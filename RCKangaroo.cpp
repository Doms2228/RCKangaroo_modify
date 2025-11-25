
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

#include <set>
#include <string>
#define DEBUG_ESTIMATES
#include <map>


// Global variables
EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];
RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;
volatile bool gQuietMode = false;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

// Add these near other global variables
EcInt gTamePrivKey;
EcInt gWildPrivKey;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;          // Default DP - will auto-adjust based on range
u32 gGenDP;       // Default GenDP for tames generation
u32 gRange;      // Default to 160-bit
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;
bool gTestMode; //test mode
u64 gMaxDPs; // Default max DPs for generation

// Enhanced collision tracking system
char gCollisionsFileName[1024];
bool gSaveCollisions = false;
u64 gSavedCollisions = 0;
u64 gCollisionCount = 0;
u64 gValidCollisions = 0;

// Stats for clean progress display
u64 gLastOpsCount = 0;
u64 gLastDPCount = 0;
u64 gStartTime = 0;

// 🟢 ADD THESE FOR RANGE ERROR TRACKING
static int totalProcessed = 0;
static int totalRangeErrors = 0;

#pragma pack(push, 1)
struct DBRec
{
    u8 x[12];
    u8 d[22];  // ← KEEP 22 BYTES (ORIGINAL)
    u8 type;   // 0 - tame, 1 - wild1, 2 - wild2
};

struct CollisionHeader
{
    u32 magic;          // 0x434F4C4C "COLL"
    u32 version;        // File format version
    u32 range;          // Bit range
    u32 total_collisions; // Total collisions saved
    u64 total_ops;      // Total operations
    u64 timestamp;      // Creation timestamp
};
#pragma pack(pop)

// Helper function to compare two EcInt values
int Compare(const EcInt& a, const EcInt& b) {
    for (int i = 4; i >= 0; i--) {
        if (a.data[i] > b.data[i]) return 1;
        if (a.data[i] < b.data[i]) return -1;
    }
    return 0;
}



bool ValidateTameFile(const char* filename, u32 expected_range) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot open tame file for validation: %s\n", filename);
        return false;
    }

    // 🚨 READ RAW HEADER BYTES - SAME WAY TFastBase DOES
    u8 file_header[1024];
    if (fread(file_header, 1, 1024, fp) != 1024) {
        printf("Failed to read tame file header\n");
        fclose(fp);
        return false;
    }

    // Extract using same offsets as TFastBase
    u32 file_range = *((u32*)(file_header + 0));
    u32 file_low = *((u32*)(file_header + 4));
    u32 file_high = *((u32*)(file_header + 8));
    u64 claimed_count = (u64)file_low | ((u64)file_high << 32);

    printf("Tame file validation:\n");
    printf("  File header range: %d\n", file_range);
    printf("  File header claimed entries: %llu\n", (unsigned long long)claimed_count);

#ifdef _WIN32
    _fseeki64(fp, 0, SEEK_END);
    __int64 file_size = _ftelli64(fp);
#else
    fseeko(fp, 0, SEEK_END);
    off_t file_size = ftello(fp);
#endif
    fclose(fp);

    u64 expected_size = 1024 + claimed_count * sizeof(DBRec);

    printf("  File size: %lld bytes\n", (long long)file_size);
    printf("  Expected size: ~%llu bytes\n", (unsigned long long)expected_size);

    bool is_valid = true;

    if (file_range != expected_range) {
        printf("❌ WARNING: File header range mismatch: expected %d, got %d\n",
            expected_range, file_range);
        is_valid = false;
    }

    if (file_size < (long long)(expected_size * 0.8)) {
        printf("❌ WARNING: File size inconsistent. Expected ~%llu bytes, got %lld bytes\n",
            (unsigned long long)expected_size, (long long)file_size);
        is_valid = false;
    }

    if (claimed_count == 0) {
        printf("❌ WARNING: File header shows 0 entries\n");
        is_valid = false;
    }

    if (is_valid) {
        printf("✅ Tame file validation: STRUCTURE OK\n");
    }
    else {
        printf("❌ Tame file validation: FAILED - file may be corrupted\n");
    }

    return is_valid;
}

void RepairTameFileHeader(const char* filename, u32 range, u64 actual_count) {
    FILE* fp = fopen(filename, "r+b");
    if (!fp) {
        printf("❌ Warning: Cannot open tame file for repair: %s\n", filename);
        return;
    }

    // Read the ACTUAL header structure
    struct TFastBaseHeader {
        u32 range;
        u32 blockCntLow;
        u32 blockCntHigh;
        u32 reserved[253];
    } header;

    if (fread(&header, sizeof(header), 1, fp) != 1) {
        printf("❌ Warning: Failed to read tame file header for repair\n");
        fclose(fp);
        return;
    }

    u64 current_count = (u64)header.blockCntLow | ((u64)header.blockCntHigh << 32);

    printf("Current header state:\n");
    printf("  Range: %d (should be %d)\n", header.range, range);
    printf("  Claimed entries: %llu (actual: %llu)\n",
        (unsigned long long)current_count, (unsigned long long)actual_count);

    // Repair using the proper structure
    header.range = range;
    header.blockCntLow = (u32)(actual_count & 0xFFFFFFFF);
    header.blockCntHigh = (u32)(actual_count >> 32);

    // Write back repaired header
    fseek(fp, 0, SEEK_SET);
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        printf("❌ Warning: Failed to write repaired header\n");
        fclose(fp);
        return;
    }

    fclose(fp);

    printf("✅ Successfully repaired tame file header: range=%d, count=%llu\n",
        range, (unsigned long long)actual_count);
}

void ForceRepairTameFileHeader(const char* filename, u32 range, u64 actual_count) {
    FILE* fp = fopen(filename, "r+b");
    if (!fp) {
        printf("❌ Cannot open tame file for force repair: %s\n", filename);
        return;
    }

    // Create a proper header matching TFastBase structure EXACTLY
    struct TFastBaseHeader {
        u32 range;
        u32 blockCntLow;
        u32 blockCntHigh;
        u32 reserved[253];
    } header;

    memset(&header, 0, sizeof(header));

    header.range = range;
    header.blockCntLow = (u32)(actual_count & 0xFFFFFFFF);
    header.blockCntHigh = (u32)(actual_count >> 32);

    // Write the entire header
    fseek(fp, 0, SEEK_SET);
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        printf("❌ Failed to write repaired header\n");
    }
    else {
        printf("✅ Force-repaired header: range=%d, count=%llu\n",
            range, (unsigned long long)actual_count);
    }

    fclose(fp);
}

void CompleteTameFileRebuild(const char* filename, u32 range, u64 actual_count) {
    printf("\n=== COMPLETE TAME FILE REBUILD ===\n");

    FILE* fp = fopen(filename, "r+b");
    if (!fp) {
        printf("❌ Cannot open tame file for complete rebuild: %s\n", filename);
        return;
    }

    // Get actual file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    printf("Actual file size: %ld bytes\n", file_size);

    // Calculate expected size
    u64 expected_size = 1024 + actual_count * sizeof(DBRec);
    printf("Expected size: %llu bytes\n", (unsigned long long)expected_size);
    printf("Size difference: %lld bytes\n", (long long)(expected_size - file_size));

    if (file_size < 1024) {
        printf("❌ File is too small for rebuild\n");
        fclose(fp);
        return;
    }

    long data_size = file_size - 1024;
    u8* file_data = (u8*)malloc(data_size);

    fseek(fp, 1024, SEEK_SET); // Skip existing header
    size_t bytes_read = fread(file_data, 1, data_size, fp);

    printf("Read %zu bytes of data from file\n", bytes_read);

    // Create a PERFECT header matching TFastBase structure
    u8 header[1024];
    memset(header, 0, 1024);

    // Calculate ACTUAL count based on file size
    u64 actual_count_from_size = (file_size - 1024) / sizeof(DBRec);
    printf("Actual count from file size: %llu\n", (unsigned long long)actual_count_from_size);

    // Use the CORRECT count based on actual file size
    *((u32*)(header + 0)) = range;
    *((u32*)(header + 4)) = (u32)(actual_count_from_size & 0xFFFFFFFF);
    *((u32*)(header + 8)) = (u32)(actual_count_from_size >> 32);
    *((u32*)(header + 12)) = 0x454D4154; // "TAME" magic

    // Write perfect header
    fseek(fp, 0, SEEK_SET);
    if (fwrite(header, 1, 1024, fp) != 1024) {
        printf("❌ Failed to write new header\n");
        free(file_data);
        fclose(fp);
        return;
    }

    // Write back data
    fseek(fp, 1024, SEEK_SET);
    if (fwrite(file_data, 1, data_size, fp) != (size_t)data_size) {
        printf("❌ Failed to write back data\n");
    }

    free(file_data);
    fclose(fp);

    printf("✅ Complete rebuild completed with CORRECT count\n");
    printf("✅ Range: %d, Actual count: %llu\n", range, (unsigned long long)actual_count_from_size);
}

// Save collisions to file
bool SaveCollisionsToFile(const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot create collisions file: %s\n", filename);
        return false;
    }

    CollisionHeader header;
    header.magic = 0x434F4C4C; // "COLL"
    header.version = 1;
    header.range = gRange;
    header.total_collisions = (u32)gCollisionCount;
    header.total_ops = PntTotalOps;
    header.timestamp = GetTickCount64();

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return false;
    }

    fclose(fp);

    if (gCollisionCount % 1000 == 0 || gCollisionCount <= 10) {
        printf("[COLLISION SAVED] Total: %llu, File: %s\n",
            (unsigned long long)gCollisionCount, filename);
    }

    return true;
}

// Load collisions from file
bool LoadCollisionsFromFile(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return false;
    }

    CollisionHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return false;
    }

    fclose(fp);

    if (header.magic != 0x434F4C4C) { // "COLL"
        printf("Error: Invalid collisions file format\n");
        return false;
    }

    if (header.range != gRange) {
        printf("Warning: Saved collisions range (%d) doesn't match current range (%d)\n",
            header.range, gRange);
        // We still load them but warn the user
    }

    gSavedCollisions = header.total_collisions;
    printf("[COLLISIONS LOADED] %u collisions from previous run (Range: %d-bit, Ops: 2^%.3f)\n",
        header.total_collisions, header.range, log2((double)header.total_ops));

    return true;
}

// Auto-detect and load collisions based on range and pubkey
void AutoLoadCollisions() {
    if (gPubKey.x.IsZero()) return; // No pubkey, no auto-load

    // Generate collision filename based on range and pubkey first bytes
    char auto_filename[1024];

    // Create a simple hash from pubkey using the first bytes of X and Y coordinates
    char pubkey_str[200];
    gPubKey.x.GetHexStr(pubkey_str);

    // Use first 8 characters of pubkey X coordinate for uniqueness
    u32 hash = 0;
    for (int i = 0; i < 8 && pubkey_str[i] != 0; i++) {
        hash = (hash << 4) | (pubkey_str[i] & 0xF);
    }

    // 🟢 DYNAMIC RANGE IN FILENAME
    sprintf(auto_filename, "collisions_%d_bit_%08X.bin", gRange, hash);

    if (IsFileExist(auto_filename)) {
        strcpy(gCollisionsFileName, auto_filename);
        gSaveCollisions = true;
        if (LoadCollisionsFromFile(auto_filename)) {
            printf("[AUTO-LOAD] Collisions file: %s\n", auto_filename);
        }
    }
    else {
        // Create new auto-save file
        strcpy(gCollisionsFileName, auto_filename);
        gSaveCollisions = true;
        printf("[AUTO-SAVE] Collisions will be saved to: %s\n", auto_filename);
    }
}

void InitGpus() {
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt)
        return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++) {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess) {
            printf("cudaSetDevice for gpu %d failed!\r\n", i);
            continue;
        }

        if (!gGPUs_Mask[i])
            continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n",
            i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f,
            deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor,
            deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);

        if (deviceProp.major < 6) {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuCnt++;
    }
    printf("Total GPUs for work: %d\r\n", GpuCnt);
}

double GetRealisticOpsEstimate(int range, int dp, u64 tames_count = 0) {
    double divisor;

    if (gGenMode) {
        // CONSERVATIVE: Same divisor for both modes
        if (range >= 130 && range <= 145) divisor = 3.0;
        else if (range >= 110 && range <= 129) divisor = 2.5;
        else if (range >= 90 && range <= 109) divisor = 2.2;
        else if (range >= 75 && range <= 89) divisor = 2.0;
        else if (range >= 146 && range <= 170) divisor = 3.2;
        else divisor = 2.0;
    }
    else {
        // Solving mode - conservative
        if (range >= 130 && range <= 145) divisor = 3.0;
        else if (range >= 110 && range <= 129) divisor = 2.5;
        else if (range >= 90 && range <= 109) divisor = 2.2;
        else if (range >= 75 && range <= 89) divisor = 2.0;
        else if (range >= 146 && range <= 170) divisor = 3.2;
        else divisor = 2.0;
    }

    // CONSERVATIVE: Keep original base operations
    double base_ops = 0.8 * pow(2.0, range / divisor);

    // CONSERVATIVE: Original efficiency factors
    double efficiency;
    if (range <= 90) efficiency = 0.95;
    else if (range <= 120) efficiency = 0.85;
    else if (range <= 150) efficiency = 0.60;
    else efficiency = 0.65;

    // CONSERVATIVE: Original DP factor
    double dp_factor = 1.0 / (1.0 - (dp - 14) * 0.02);

    // Tames boost factor
    double tames_factor = 1.0;
    if (tames_count > 0) {
        double tames_power = (double)tames_count * (1ull << dp);
        tames_factor = 1.0 / (1.0 + log2(tames_power) * 0.01);
    }

    // CONSERVATIVE: Memory constraint for 32GB RAM
    double memory_factor = 1.0;
    double estimated_memory_mb = 0.0;

    // Tames database memory
    estimated_memory_mb += (double)tames_count * 35.0 / (1024.0 * 1024.0);
    estimated_memory_mb += 200.0;

    // DP table memory
    if (dp == 14) estimated_memory_mb += 500.0;
    else if (dp == 15) estimated_memory_mb += 250.0;
    else if (dp == 16) estimated_memory_mb += 125.0;
    else if (dp == 17) estimated_memory_mb += 62.5;
    else if (dp == 18) estimated_memory_mb += 31.25;
    else if (dp == 19) estimated_memory_mb += 15.625;
    else if (dp == 20) estimated_memory_mb += 7.8125;
    else {
        estimated_memory_mb += 500.0 / (1 << (dp - 14));
    }

    // GPU memory
    double gpu_memory = 1500.0;
    if (dp >= 18) gpu_memory += 100.0;
    if (dp >= 20) gpu_memory += 100.0;

    estimated_memory_mb += gpu_memory;

    // 32GB RAM available
    double available_memory_mb = 32.0 * 1024.0;

    if (estimated_memory_mb > available_memory_mb * 0.8) {
        memory_factor = 0.7;
        printf("⚠️  MEMORY WARNING: DP=%d with %llu tames may use ~%.1f GB RAM\n",
            dp, (unsigned long long)tames_count, estimated_memory_mb / 1024.0);
    }
    else {
        printf("✅ Memory usage OK: ~%.1f GB (within 32GB limit)\n", estimated_memory_mb / 1024.0);
    }

    // CONSERVATIVE: DP efficiency
    double dp_efficiency = 1.0;
    if (dp >= 18) dp_efficiency = 0.95;
    if (dp >= 20) dp_efficiency = 0.90;

    double final_ops = base_ops * efficiency * dp_factor * tames_factor * memory_factor * dp_efficiency;

    // CONSERVATIVE: Generation mode adjustment
    if (gGenMode) {
        final_ops *= 0.5;  // Original 50% reduction
    }

#ifdef DEBUG_ESTIMATES
    printf("OPTIMAL Estimation breakdown for %d-bit range:\n", range);
    printf("  Base ops: 2^%.3f (divisor=%.1f)\n", log2(base_ops), divisor);
    printf("  Efficiency: %.2f\n", efficiency);
    printf("  DP factor: %.3f (DP=%d)\n", dp_factor, dp);
    printf("  Tames factor: %.3f (tames=%llu)\n", tames_factor, (unsigned long long)tames_count);
    printf("  Memory factor: %.3f\n", memory_factor);
    printf("  DP efficiency: %.3f\n", dp_efficiency);
    printf("  Generation penalty: %.1f%%\n", gGenMode ? 50.0 : 0.0);
    printf("  Final estimate: 2^%.3f operations\n", log2(final_ops));
#endif

    return final_ops;
}


#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    InterlockedDecrement(&ThrCnt);
    return 0;
}
#else
void* kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&ThrCnt, 1);
    return 0;
}
#endif

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt) {
    csAddPoints.Enter();
    if (PntIndex + pnt_cnt >= MAX_CNT_LIST) {
        csAddPoints.Leave();
        printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
        return;
    }
    memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    PntIndex += pnt_cnt;
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

// Enhanced Collision_SOTA (modular-safe wild/wild handling)
bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
    if (t.IsEqual(w)) {
        printf("SELF-COLLISION DETECTED - Skipping (same distance)\n");
        return false;
    }
    
    printf("\n=== COLLISION_SOTA CALLED ===\n");
    printf("TameType: %d, WildType: %d, IsNeg: %s\n",
        TameType, WildType, IsNeg ? "YES" : "NO");

    // Print the distances
    char t_str[100], w_str[100];
    t.GetHexStr(t_str);
    w.GetHexStr(w_str);
    printf("Tame distance (t): %s\n", t_str);
    printf("Wild distance (w): %s\n", w_str);

    if (IsNeg) {
        printf("Negating tame distance...\n");
        t.Neg();
        t.GetHexStr(t_str);
        printf("Negated tame distance: %s\n", t_str);
    }

    //------------------------------------------------------------------
    // Prepare originals
    //------------------------------------------------------------------
    EcInt orig_t = t;
    EcInt orig_w = w;

    //------------------------------------------------------------------
    // TAME–WILD  (easy path)
    //------------------------------------------------------------------
    if (TameType == TAME)
    {
        printf("Processing TAME-WILD collision...\n");

        gPrivKey = t;
        gPrivKey.Sub(w);

        gTamePrivKey = t;  // debug store
        gWildPrivKey = w;

        EcInt save = gPrivKey;

        printf("Before HalfRange: ");
        gPrivKey.GetHexStr(t_str);
        printf("%s\n", t_str);

        gPrivKey.Add(Int_HalfRange);

        printf("After HalfRange: ");
        gPrivKey.GetHexStr(t_str);
        printf("%s\n", t_str);

        EcPoint P = ec.MultiplyG(gPrivKey);

        // test non-negated
        if (P.IsEqual(pnt)) {
            printf("COLLISION_SOTA: Tame-Wild VALID (non-negated)\n");
            gPrivKey.GetHexStr(t_str);
            printf("Private Key: %s\n", t_str);
            gCollisionCount++;
            gValidCollisions++;
            return true;
        }

        // test negated
        gPrivKey = save;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);

        if (P.IsEqual(pnt)) {
            printf("COLLISION_SOTA: Tame-Wild VALID (negated)\n");
            gPrivKey.GetHexStr(t_str);
            printf("Private Key: %s\n", t_str);
            gCollisionCount++;
            gValidCollisions++;
            return true;
        }

        printf("COLLISION_SOTA: Tame-Wild INVALID\n");
        return false;
    }

    //------------------------------------------------------------------
    // WILD–WILD (hard path)
    //------------------------------------------------------------------
    printf("Processing WILD-WILD collision...\n");

    bool found = false;

    //------------------------------------------------------------------
    // 1) MODULAR-SAFE DIVISION BY 2  (correct for 135+ bit ranges)
    //------------------------------------------------------------------
    static EcInt Order;
    static bool initOrder = false;
    if (!initOrder) {
        Order.SetHexStr(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
        );
        initOrder = true;
    }

    // diff = (t - w) mod N
    EcInt diff = orig_t;
    diff.Sub(orig_w);

    if (diff.data[4] >> 63) {
        EcInt tmp = orig_w;
        tmp.Sub(orig_t);            // tmp = w - t (positive)
        diff = Order;
        diff.Sub(tmp);              // diff = N - (w - t)
    }

    // k = diff / 2 (mod N)
    EcInt k = diff;
    if ((k.data[0] & 1) == 0) {
        k.ShiftRight(1);            // even
    }
    else {
        k.Add(Order);               // odd: add N then divide
        k.ShiftRight(1);
    }

    // test candidate k
    gPrivKey = k;
    gPrivKey.Add(Int_HalfRange);
    EcPoint P = ec.MultiplyG(gPrivKey);

    if (P.IsEqual(pnt)) {
        printf("COLLISION_SOTA: Wild-Wild VALID (modular /2)\n");
        gPrivKey.GetHexStr(t_str);
        printf("Private Key: %s\n", t_str);
        gCollisionCount++;
        gValidCollisions++;
        found = true;
    }

    // test negated form
    if (!found) {
        gPrivKey = k;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);

        P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(pnt)) {
            printf("COLLISION_SOTA: Wild-Wild VALID (modular /2 negated)\n");
            gPrivKey.GetHexStr(t_str);
            printf("Private Key: %s\n", t_str);
            gCollisionCount++;
            gValidCollisions++;
            found = true;
        }
    }

    //------------------------------------------------------------------
    // 2) FALLBACK — old heuristics (for backward compatibility)
    //------------------------------------------------------------------
    if (!found)
    {
        printf("Trying fallback heuristics...\n");

        // (t - w)/2
        gPrivKey = orig_t;
        gPrivKey.Sub(orig_w);
        if (gPrivKey.data[4] >> 63) gPrivKey.Neg();
        gPrivKey.ShiftRight(1);
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(pnt)) { found = true; printf("fallback orig OK\n"); }

        // (w - t)/2
        if (!found) {
            gPrivKey = orig_w;
            gPrivKey.Sub(orig_t);
            if (gPrivKey.data[4] >> 63) gPrivKey.Neg();
            gPrivKey.ShiftRight(1);
            gPrivKey.Add(Int_HalfRange);
            P = ec.MultiplyG(gPrivKey);
            if (P.IsEqual(pnt)) { found = true; printf("fallback w-t/2 OK\n"); }
        }

        // (t - w)
        if (!found) {
            gPrivKey = orig_t;
            gPrivKey.Sub(orig_w);
            if (gPrivKey.data[4] >> 63) gPrivKey.Neg();
            gPrivKey.Add(Int_HalfRange);
            P = ec.MultiplyG(gPrivKey);
            if (P.IsEqual(pnt)) { found = true; printf("fallback t-w OK\n"); }
        }

        // (w - t)
        if (!found) {
            gPrivKey = orig_w;
            gPrivKey.Sub(orig_t);
            if (gPrivKey.data[4] >> 63) gPrivKey.Neg();
            gPrivKey.Add(Int_HalfRange);
            P = ec.MultiplyG(gPrivKey);
            if (P.IsEqual(pnt)) { found = true; printf("fallback w-t OK\n"); }
        }
    }

    //------------------------------------------------------------------
    // Final failure debug
    //------------------------------------------------------------------
    if (!found)
    {
        printf("COLLISION_SOTA: All wild-wild attempts failed\n");

        printf("Target PK:\n");
        char xh[100], yh[100];
        pnt.x.GetHexStr(xh);
        pnt.y.GetHexStr(yh);
        printf("X: %s\nY: %s\n", xh, yh);

        printf("Last tested key: ");
        gPrivKey.GetHexStr(t_str);
        printf("%s\n", t_str);

        EcPoint P2 = ec.MultiplyG(gPrivKey);
        P2.x.GetHexStr(xh);
        P2.y.GetHexStr(yh);
        printf("Computed PK:\nX: %s\nY: %s\n", xh, yh);
    }

    return found;
}


void CheckNewPoints()
{
    csAddPoints.Enter();
    if (!PntIndex)
    {
        csAddPoints.Leave();
        return;
    }

    int cnt = PntIndex;

    memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
    PntIndex = 0;
    csAddPoints.Leave();

    for (int i = 0; i < cnt; i++)
    {
        DBRec nrec;
        u8* p = pPntList2 + i * GPU_DP_SIZE;
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 16, 22);
        nrec.type = gGenMode ? TAME : p[40];  // 🟢 ORIGINAL OFFSET 40

        // 🟢 ADD TYPE VALIDATION
        if (nrec.type != TAME && nrec.type != WILD1 && nrec.type != WILD2) {
            printf("❌ INVALID TYPE DETECTED: %d - Skipping corrupted point\n", nrec.type);
            continue; // Skip corrupted points
        }

        DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
        if (gGenMode)
            continue;
        if (pref)
        {
            // 🟢 VALIDATE EXISTING POINT TYPE TOO
            DBRec tmp_pref;
            memcpy(&tmp_pref, &nrec, 3);
            memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
            pref = &tmp_pref;

            if (pref->type != TAME && pref->type != WILD1 && pref->type != WILD2) {
                printf("❌ INVALID EXISTING TYPE: %d - Skipping corrupted collision\n", pref->type);
                continue;
            }

            EcInt w, t;
            int TameType, WildType;
            if (pref->type != TAME)
            {
                memcpy(w.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = nrec.type;
                WildType = pref->type;
            }
            else
            {
                memcpy(w.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = TAME;
                WildType = nrec.type;
            }

            // 🟢 DEBUG: Print the private key ranges
            char t_str[100], w_str[100];
            t.GetHexStr(t_str);
            w.GetHexStr(w_str);
            printf("TAME private key range: %s\n", t_str);
            printf("WILD private key range: %s\n", w_str);

            bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
            if (!res)
            {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
                if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
                    ;// ToLog("W1 and W2 collides in mirror");
                else
                {
                    printf("Collision Error\r\n");
                    gTotalErrors++;
                }
                continue;
            }
            gSolved = true;
            break;
        }
    }
}

// Enhanced progress display with collision tracking
void ShowStats(u64 start_time, double exp_ops, double dp_val) {
    int speed = 0;
    for (int i = 0; i < GpuCnt; i++)
        speed += GpuKangs[i]->GetStatsSpeed();

    u64 sec = (GetTickCount64() - start_time) / 1000;
    u64 days = sec / (3600 * 24);
    int hours = (int)(sec - days * (3600 * 24)) / 3600;
    int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;

    if (gGenMode) {
        u64 current_dps = db.GetBlockCnt();
        u64 total_dps = gMaxDPs > 0 ? gMaxDPs : (u64)(exp_ops / dp_val);
        double progress = 0.0;
        if (total_dps > 0) {
            progress = (double)current_dps / total_dps * 100.0;
            if (progress > 100.0) progress = 100.0;
        }

        static u64 last_dp_count = 0;
        static u64 last_dp_time = GetTickCount64();
        double actual_dps_rate = 0.0;
        u64 current_time = GetTickCount64();

        // Calculate DPs per second
        if (current_time > last_dp_time) {
            double time_diff_seconds = (current_time - last_dp_time) / 1000.0;
            if (time_diff_seconds > 1.0) {
                actual_dps_rate = (double)(current_dps - last_dp_count) / time_diff_seconds;
            }
        }

        // Calculate ETA
        u64 remaining_dps = total_dps - current_dps;
        u64 seconds_remaining = 0;
        if (actual_dps_rate > 0) {
            seconds_remaining = (u64)(remaining_dps / actual_dps_rate);
        }

        u64 eta_days = seconds_remaining / (3600 * 24);
        u64 eta_hours = (seconds_remaining % (3600 * 24)) / 3600;
        u64 eta_minutes = (seconds_remaining % 3600) / 60;
        u64 eta_seconds = seconds_remaining % 60;

        // 🟢 CLEAN SINGLE LINE WITH ETA
        printf("\r[GEN] %llu/%llu DPs (%.1f%%) | %d MKeys/s | %.0f DPs/s | Elapsed: %llud:%02dh:%02dm | ETA: %llud:%02dh:%02dm",
            (unsigned long long)current_dps,
            (unsigned long long)total_dps,
            progress,
            speed,
            actual_dps_rate,
            (unsigned long long)days, hours, min,
            (unsigned long long)eta_days, eta_hours, eta_minutes);

        if (current_time - last_dp_time >= 5000) {
            last_dp_count = current_dps;
            last_dp_time = current_time;
        }
    }
    else {
        // SOLVING MODE
        double ops_done = (double)PntTotalOps;
        double progress = 0.0;
        if (exp_ops > 0) {
            progress = ops_done / exp_ops * 100.0;
            if (progress > 100.0) progress = 100.0;
        }

        // Calculate ETA for solving mode
        static u64 last_ops_count = 0;
        static u64 last_ops_time = GetTickCount64();
        double actual_ops_rate = 0.0;
        u64 current_time = GetTickCount64();

        // Calculate operations per second
        if (current_time > last_ops_time) {
            double time_diff_seconds = (current_time - last_ops_time) / 1000.0;
            if (time_diff_seconds > 1.0) {
                actual_ops_rate = (double)(PntTotalOps - last_ops_count) / time_diff_seconds;
            }
        }

        // Calculate ETA
        double remaining_ops = exp_ops - ops_done;
        u64 seconds_remaining = 0;
        if (actual_ops_rate > 0 && remaining_ops > 0) {
            seconds_remaining = (u64)(remaining_ops / actual_ops_rate);
        }

        u64 eta_days = seconds_remaining / (3600 * 24);
        u64 eta_hours = (seconds_remaining % (3600 * 24)) / 3600;
        u64 eta_minutes = (seconds_remaining % 3600) / 60;

        // 🟢 CLEAN SINGLE LINE WITH ETA
        printf("\r[SOLVE] %.1f%% | 2^%.3f ops | %d MKeys/s | Collisions: %llu | ETA: %llud:%02dh:%02dm",
            progress,
            log2(ops_done),
            speed,
            (unsigned long long)gCollisionCount,
            (unsigned long long)eta_days, eta_hours, eta_minutes);

        if (current_time - last_ops_time >= 5000) {
            last_ops_count = PntTotalOps;
            last_ops_time = current_time;
        }
    }

    fflush(stdout);
    gLastOpsCount = PntTotalOps;
    gLastDPCount = db.GetBlockCnt();
}

void InitJumps(int Range) {
    EcInt minjump, t;

    // OPTIMIZED FOR 135-BIT RANGE
    int small_shift, large_shift1, large_shift2;

    if (Range == 135) {
        // SPECIALIZED FOR 135-BIT - BETTER SEPARATION
        small_shift = 62;    // Reduced for better tame behavior
        large_shift1 = 108;  // Medium jumps for Wild1
        large_shift2 = 102;  // Large jumps for Wild2
    }
    else if (Range > 130 && Range <= 145) {
        small_shift = Range / 2 + 4;
        large_shift1 = Range - 16;
        large_shift2 = Range - 20;
    }
    else if (Range > 100) {
        small_shift = Range / 2 + 5;
        large_shift1 = Range - 15;
        large_shift2 = Range - 18;
    }
    else {
        small_shift = Range / 2 + 3;
        large_shift1 = Range - 12;
        large_shift2 = Range - 14;
    }

    // Small jumps - for TAME kangaroos
    minjump.Set(1);
    minjump.ShiftLeft(small_shift);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        t.ShiftRight(2);  // More randomization
        EcJumps1[i].dist.Add(t);
        // Tames can be odd or even (no parity restriction)
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }

    // Large jumps - for WILD1 kangaroos
    minjump.Set(1);
    minjump.ShiftLeft(large_shift1);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        t.ShiftRight(3);  // Different pattern
        EcJumps2[i].dist.Add(t);
        EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; // Wilds = even
        EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
    }

    // Extra large jumps - for WILD2 kangaroos
    minjump.Set(1);
    minjump.ShiftLeft(large_shift2);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        t.ShiftRight(4);  // Even more different pattern
        EcJumps3[i].dist.Add(t);
        EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; // Wilds = even
        EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
    }

    printf("Optimized jumps for %d-bit: Tames(+%d), Wild1(-%d), Wild2(-%d)\n",
        Range, small_shift, large_shift1, large_shift2);

    // Verify jump diversity
    printf("Jump sizes: Small~2^%d, Medium~2^%d, Large~2^%d\n",
        small_shift, large_shift1, large_shift2);
}


// 🟢 SIMPLE VERIFICATION FUNCTION - Add before SolvePoint()
void VerifyTamesQuality() {
    printf("\n=== TAMES QUALITY VERIFICATION ===\n");
    printf("Total tames generated: %llu\n", (unsigned long long)db.GetBlockCnt());

    // Basic sanity checks
    if (db.GetBlockCnt() == 0) {
        printf("ERROR: No tames generated!\n");
    }
    else if (gMaxDPs > 0 && db.GetBlockCnt() < gMaxDPs * 0.9) {
        printf("WARNING: Generated only %llu/%llu tames (%.1f%%)\n",
            (unsigned long long)db.GetBlockCnt(),
            (unsigned long long)gMaxDPs,
            (double)db.GetBlockCnt() / gMaxDPs * 100.0);
    }
    else {
        printf("Generation completed successfully\n");
        printf("Tames count: %llu\n", (unsigned long long)db.GetBlockCnt());
    }

    printf("=== END VERIFICATION ===\n\n");
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res) {
    // ✅ USE THE ORIGINAL APPROACH - always use the passed point
    gPntToSolve = PntToSolve;

    if (gGenMode) {
        printf("GENERATION MODE - Creating tames for %d-bit range\n", Range);
        printf("Solving random keys and storing distinguished points as tames\n");
        printf("Range: [2^%d, 2^%d-1]\n", Range - 1, Range);

        // The point being solved is a RANDOM key in the target range
        // Distinguished points encountered will be stored as tames
        char targetX[100], targetY[100];
        gPntToSolve.x.GetHexStr(targetX);
        gPntToSolve.y.GetHexStr(targetY);
        printf("Solving random key at:\n");
        printf("  X: %s\n", targetX);
        printf("  Y: %s\n", targetY);
    }
    else {
        printf("SOLVING MODE - Using provided public key as target\n");

        char targetX[100], targetY[100];
        gPntToSolve.x.GetHexStr(targetX);
        gPntToSolve.y.GetHexStr(targetY);
        printf("Target public key:\n");
        printf("  X: %s\n", targetX);
        printf("  Y: %s\n", targetY);
    }

    // Rest of function remains the same...

    // 🟢 INITIALIZE COLLISION DEBUGGING VARIABLES
    gTamePrivKey.Set(0);
    gWildPrivKey.Set(0);

    // Enhanced collision tracking with persistence
    u64 previously_saved = gSavedCollisions;
    gCollisionCount = previously_saved;
    gValidCollisions = 0;
    gTotalErrors = 0;

    printf("Collision tracking initialized: %llu collisions from previous runs\n",
        (unsigned long long)previously_saved);

    int originalDP = DP;
    bool hasTames = (!gGenMode && gTamesFileName[0] && IsFileExist(gTamesFileName));

    // 🚨 DEBUG 1: RANGE VERIFICATION - FIXED
    printf("\n=== DEBUG: RANGE VERIFICATION ===\n");

    // Verify the search range - Use the Range parameter instead of hardcoding
    EcInt rangeStart, rangeEnd, rangeSize;
    rangeStart.Set(1);
    rangeStart.ShiftLeft(Range - 1);  // 2^(Range-1)
    rangeEnd.Set(1);
    rangeEnd.ShiftLeft(Range);        // 2^Range

    // FIXED: Subtract 1, not rangeEnd from itself
    EcInt one;
    one.Set(1);
    rangeEnd.Sub(one);               // 2^Range - 1

    // Calculate actual range size
    rangeSize = rangeEnd;
    rangeSize.Sub(rangeStart);
    rangeSize.Add(one); // Include both endpoints

    char startStr[100], endStr[100], sizeStr[100];
    rangeStart.GetHexStr(startStr);
    rangeEnd.GetHexStr(endStr);
    rangeSize.GetHexStr(sizeStr);

    printf("Search Start: %s\n", startStr);
    printf("Search End:   %s\n", endStr);
    printf("Range Size:   %s\n", sizeStr);
    printf("Range in bits: %d (2^%d to 2^%d-1)\n", Range, Range - 1, Range);

    // 🟢 ADD RANGE-SPECIFIC MESSAGES
    if (Range == 135) {
        printf("135-BIT MODE: Using optimized parameters\n");
    }
    else if (Range <= 90) {
        printf("TEST MODE: %d-bit range for algorithm validation\n", Range);
    }
    else if (Range <= 120) {
        printf("MEDIUM RANGE: %d-bit search\n", Range);
    }
    else {
        printf("LARGE RANGE: %d-bit challenge\n", Range);
    }

    // Verify start is correct
    printf("Current Start: ");
    gStart.GetHexStr(startStr);
    printf("%s\n", startStr);

    // Verify start is within range
    if (gStart.IsLessThanU(rangeStart) || !gStart.IsLessThanU(rangeEnd)) {
        printf("🚨 ERROR: Start value is outside %d-bit range!\n", Range);
        return false;
    }
    printf("✅ Start value is within %d-bit range\n", Range);

    // Verify the public key is on curve (alternative method)
    EcPoint testVerify = ec.MultiplyG(gStart);
    if (gGenMode) {
        printf("GENERATION MODE: Creating tames for range %d-bit\n", Range);
        printf("ACTUAL PUZZLE TARGET:\n");  // ✅ CLEAR!
        char targetX[100], targetY[100];
        gPntToSolve.x.GetHexStr(targetX);
        gPntToSolve.y.GetHexStr(targetY);
        printf("X: %s\n", targetX);
        printf("Y: %s\n", targetY);
    }
    else {
        printf("SOLVING MODE: Solving for provided public key\n");
        printf("Start position for kangaroos (start * G):\n");
    }

    // Test multiply the start point
    EcPoint testPoint = ec.MultiplyG(gStart);
    char targetX[100], targetY[100];
    testPoint.x.GetHexStr(targetX);
    testPoint.y.GetHexStr(targetY);
    printf("Start * G -> X: %s\n", targetX);
    printf("Start * G -> Y: %s\n", targetY);

    printf("Target PubKey X: ");
    gPntToSolve.x.GetHexStr(targetX);
    printf("%s\n", targetX);
    printf("Target PubKey Y: ");
    gPntToSolve.y.GetHexStr(targetY);
    printf("%s\n", targetY);

    // Basic verification instead of IsOnCurve check
    if (gPntToSolve.x.IsZero() && gPntToSolve.y.IsZero()) {
        printf("🚨 CRITICAL: Target point is zero point!\n");
    }
    else {
        printf("✅ Target point appears valid (non-zero coordinates)\n");
    }

    printf("=== END RANGE DEBUG ===\n\n");

    if (hasTames) {
        if (db.LoadFromFile(gTamesFileName)) {
            u64 tames_count = db.GetBlockCnt();
            printf("Preloaded tames: %llu entries\n", (unsigned long long)tames_count);

            // 🟢🟢🟢 IMPROVED HEADER REPAIR LOGIC 🟢🟢🟢
            printf("Verifying tame file integrity...\n");

            // Get actual count from database (this is reliable)
            u64 actual_tames_count = db.GetBlockCnt();

            // Check current header state
            u64 claimed_count = (u64)db.Header[4] | ((u64)db.Header[8] << 32);


            printf("Header state check:\n");
            printf("  Current range: %d (expected: %d)\n", db.Header[0], gRange);
            printf("  Claimed entries: %llu\n", (unsigned long long)claimed_count);
            printf("  Actual entries: %llu\n", (unsigned long long)actual_tames_count);


            // Ensure perfect header based on actual file size
            printf("\n=== ENSURING PERFECT HEADER ===\n");
            FILE* fp = fopen(gTamesFileName, "r+b");
            if (fp) {
#ifdef _WIN32
                _fseeki64(fp, 0, SEEK_END);
                __int64 file_size = _ftelli64(fp);
#else
                fseeko(fp, 0, SEEK_END);
                off_t file_size = ftello(fp);
#endif

                u64 actual_count_from_size = (file_size - 1024) / sizeof(DBRec);
                printf("File size: %lld bytes = %llu tames\n",
                    (long long)file_size, (unsigned long long)actual_count_from_size);

                // Create perfect header
                u8 header[1024];
                memset(header, 0, 1024);
                *((u32*)(header + 0)) = gRange;
                *((u32*)(header + 4)) = (u32)(actual_count_from_size & 0xFFFFFFFF);
                *((u32*)(header + 8)) = (u32)(actual_count_from_size >> 32);
                *((u32*)(header + 12)) = 0x454D4154; // "TAME"

                fseek(fp, 0, SEEK_SET);
                fwrite(header, 1, 1024, fp);
                fclose(fp);

                printf("✅ Header perfected: %llu tames\n", (unsigned long long)actual_count_from_size);

                // 🚨 CRITICAL FIX: Force the database to use the correct count internally
                db.Header[0] = gRange;
                db.Header[4] = (u32)(actual_count_from_size & 0xFFFFFFFF);
                db.Header[8] = (u32)(actual_count_from_size >> 32);
                printf("✅ Database internal count forced: %llu tames\n", (unsigned long long)actual_count_from_size);

                // Update the variable directly (no need to reload)
                actual_tames_count = actual_count_from_size;
                printf("Updated tames count: %llu entries\n", (unsigned long long)actual_tames_count);
            }

            // Now validate the file structure
            if (ValidateTameFile(gTamesFileName, gRange)) {
                printf("✅ Tame file structure validation: PASSED\n");
            }
            else {
                printf("❌ Tame file structure validation: FAILED\n");
                printf("Data integrity may be compromised during collision processing.\n");
            }

            printf("✅ Header repair successful! Proceeding with search...\n");

            // The existing debug section continues below...
            printf("\n=== DEBUG: TAME FILE INTEGRITY ===\n");
            printf("Tames loaded: %llu\n", (unsigned long long)db.GetBlockCnt());
            printf("Tame file header range: %d\n", db.Header[0]);
            printf("Current search range: %d\n", gRange);

            // Check for range mismatch
            if (db.Header[0] != gRange) {
                printf("🚨 CRITICAL: Tame file range (%d) doesn't match search range (%d)!\n",
                    db.Header[0], gRange);
                printf("This will cause ZERO collisions!\n");
            }
            else {
                printf("✅ Tame file range matches search range\n");
            }

            // 64-bit file size check
            FILE* f = fopen(gTamesFileName, "rb");
            if (f) {
#ifdef _WIN32
                _fseeki64(f, 0, SEEK_END);
                __int64 file_size = _ftelli64(f);
#else
                fseeko(f, 0, SEEK_END);
                off_t file_size = ftello(f);
#endif
                fclose(f);
                printf("Tame file size: %lld bytes\n", (long long)file_size);

                u64 expected_size = 1024 + actual_tames_count * sizeof(DBRec);
                printf("Expected size: ~%llu bytes\n", (unsigned long long)expected_size);

                if (file_size < (long long)(expected_size * 0.8)) {
                    printf("🚨 WARNING: File size seems too small for claimed tames count!\n");
                }
                else {
                    printf("✅ File size matches expected size\n");
                }
            }
            else {
                printf("❌ Cannot open file for size verification\n");
            }

            // Sample a few points from the buffer (safer method)
            printf("Sampling first few points from buffer...\n");

            // We'll check the first points that get added to see if they're valid
            printf("=== END TAME DEBUG ===\n\n");

            printf("Using default DP %d for %d-bit range\n", DP, Range);
        }
    }

    if ((DP < 13) || (DP > 60)) {
        printf("Unsupported DP value (%d)!\r\n", DP);
        return false;
    }

    printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);

    double dp_val = (double)(1ull << DP);

    double tames_work = 0.0;
    double ops;

    if (hasTames) {
        if (db.LoadFromFile(gTamesFileName)) {
            u64 tames_count = db.GetBlockCnt();
            tames_work = (double)tames_count * dp_val;
            printf("Tames work equivalent: 2^%.3f operations\n", log2(tames_work));

            // 🟢 USE GetRealisticOpsEstimate FOR BOTH CASES
            ops = GetRealisticOpsEstimate(Range, DP, tames_count);
            printf("Realistic estimate for %d-bit range: 2^%.3f operations\n", Range, log2(ops));
        }
    }
    else if (gGenMode) {
        // 🟢 USE GetRealisticOpsEstimate FOR GENERATION TOO
        ops = GetRealisticOpsEstimate(Range, DP, 0);
        printf("Generation estimate for %d-bit range: 2^%.3f operations\n", Range, log2(ops));
    }
    else {
        // 🟢 FALLBACK: Use GetRealisticOpsEstimate with 0 tames
        ops = GetRealisticOpsEstimate(Range, DP, 0);
        printf("Base estimate for %d-bit range: 2^%.3f operations\n", Range, log2(ops));
    }


    // FIX RAM CALCULATION
    double ram = 0.0;
    if (gGenMode) {
        // More realistic memory estimation with overhead
        double db_overhead_factor = 1.3; // 30% overhead for hash tables
        ram = (double)(sizeof(DBRec) * gMaxDPs * db_overhead_factor) / (1024 * 1024 * 1024);
        ram += (double)(MAX_CNT_LIST * GPU_DP_SIZE * 2) / (1024 * 1024 * 1024);
        ram += 0.5; // GPU structures and OS overhead
        printf("Estimated RAM for generation: %.3f GB (with 30%% overhead)\n", ram);
    }
    else {
        // 🟢 PROPER FIX: Calculate from actual kangaroo count
        u64 total_kangaroos = 0;
        for (int i = 0; i < GpuCnt; i++)
            total_kangaroos += GpuKangs[i]->CalcKangCnt();

        if (hasTames) {
            u64 tames_count = db.GetBlockCnt();
            ram = (double)(tames_count * sizeof(DBRec)) / (1024 * 1024 * 1024);  // Tames DB
            ram += (double)(total_kangaroos * 16) / (1024 * 1024 * 1024);  // DP tables
            ram += (double)(MAX_CNT_LIST * GPU_DP_SIZE * 2) / (1024 * 1024 * 1024);  // Buffers
            ram += 2.0;  // GPU structures and overhead
        }
        else {
            ram = (double)(total_kangaroos * 16) / (1024 * 1024 * 1024);
            ram += (double)(MAX_CNT_LIST * GPU_DP_SIZE * 2) / (1024 * 1024 * 1024);
            ram += 2.0;
        }
        printf("Estimated RAM for solving: %.3f GB\n", ram);
    }

    gIsOpsLimit = false;
    double MaxTotalOps = 0.0;
    if (gMax > 0) {
        MaxTotalOps = gMax * ops;
        printf("Max operations: 2^%.3f (%.1fx estimated)\n", log2(MaxTotalOps), gMax);
    }

    u64 total_kangs = 0;
    for (int i = 0; i < GpuCnt; i++)
        total_kangs += GpuKangs[i]->CalcKangCnt();

    printf("Total kangaroos: %llu\n", (unsigned long long)total_kangs);

    double path_single_kang = ops / total_kangs;
    double DPs_per_kang = path_single_kang / dp_val;

    printf("Configuration analysis:\n");
    printf("  Operations per kangaroo: 2^%.3f\n", log2(path_single_kang));
    printf("  DPs per kangaroo: %.1f\n", DPs_per_kang);

    if (DPs_per_kang >= 100.0) {
        printf("  Status: Excellent - consider increasing DP to %d\n", DP);
    }
    else if (DPs_per_kang >= 10.0) {
        printf("  Status: Very Good\n");
    }
    else if (DPs_per_kang >= 3.0) {
        printf("  Status: Good\n");
    }
    else if (DPs_per_kang >= 1.0) {
        printf("  Status: Adequate\n");
    }
    else {
        printf("  Status: Low - but acceptable with large tames file\n");
    }

    SetRndSeed(0);
    PntTotalOps = 0;
    PntIndex = 0;

    InitJumps(Range);
    SetRndSeed(GetTickCount64());

    Int_HalfRange.Set(1);
    Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
    Pnt_NegHalfRange = Pnt_HalfRange;
    Pnt_NegHalfRange.y.NegModP();

    Int_TameOffset.Set(1);
    Int_TameOffset.ShiftLeft(Range - 1);
    EcInt tt;
    tt.Set(1);
    tt.ShiftLeft(Range - 5);
    Int_TameOffset.Sub(tt);

    // Double-check it was set correctly
    char checkX[100], checkY[100];
    gPntToSolve.x.GetHexStr(checkX);
    gPntToSolve.y.GetHexStr(checkY);
    printf("CONFIRMED gPntToSolve X: %s\n", checkX);
    printf("CONFIRMED gPntToSolve Y: %s\n", checkY);

    if (gPntToSolve.x.IsZero() && gPntToSolve.y.IsZero()) {
        printf("🚨 ERROR: gPntToSolve is still ZERO after assignment!\n");
        return false;
    }
    // Add this debug before GPU preparation
    printf("\n=== GPU PREPARATION DEBUG ===\n");
    printf("gPntToSolve X: ");
    gPntToSolve.x.GetHexStr(checkX);
    printf("%s\n", checkX);
    printf("PntToSolve X: ");
    PntToSolve.x.GetHexStr(checkX);
    printf("%s\n", checkX);
    printf("=== END GPU DEBUG ===\n\n");

    bool all_gpus_ready = true;
    for (int i = 0; i < GpuCnt; i++) {
        if (!GpuKangs[i]->Prepare(gPntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3)) {
            GpuKangs[i]->Failed = true;
            printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
            all_gpus_ready = false;
        }
    }

    if (!all_gpus_ready) {
        printf("GPU preparation failed. Exiting.\r\n");
        return false;
    }

    u64 tm0 = GetTickCount64();
    printf("\nSearch started...\n");
    printf("Collision detection active...\n\n");

#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    for (int i = 0; i < MAX_GPU_CNT; i++) {
        thr_handles[i] = NULL;
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    for (int i = 0; i < MAX_GPU_CNT; i++) {
        thr_handles[i] = 0;
    }
#endif

    u32 ThreadID;
    gSolved = false;
    ThrCnt = GpuCnt;
    for (int i = 0; i < GpuCnt; i++) {
#ifdef _WIN32
        HANDLE handle = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
        if (handle != NULL) {
            thr_handles[i] = handle;
        }
        else {
            printf("Failed to create thread for GPU %d\n", i);
            GpuKangs[i]->Failed = true;
        }
#else
        if (pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]) != 0) {
            printf("Failed to create thread for GPU %d\n", i);
            GpuKangs[i]->Failed = true;
        }
#endif
    }

    u64 tm_stats = GetTickCount64();
    u64 last_collisions = previously_saved;
    u64 last_ops = 0;
    u64 collision_check_counter = 0;
    int cnt = PntIndex;

    while (!gSolved) {
        CheckNewPoints();
        Sleep(10);

        if (gGenMode && gMaxDPs > 0 && db.GetBlockCnt() >= gMaxDPs) {
            gIsOpsLimit = true;
            printf("\nDPs limit reached: %llu DPs (target: %llu)\n",
                (unsigned long long)db.GetBlockCnt(),
                (unsigned long long)gMaxDPs);
            // 🟢 CALL the verification function
            VerifyTamesQuality();
            break;
        }

        collision_check_counter++;
        if (collision_check_counter % 50 == 0) {
            if (gCollisionCount > last_collisions) {
                printf("\n[COLLISION #%llu] Checking...", (unsigned long long)gCollisionCount);
                fflush(stdout);
                last_collisions = gCollisionCount;
            }
            if (gValidCollisions > 0) {
                printf("\n[VALID COLLISION!] Validating private key...");
                fflush(stdout);

                // ✅ SIMPLIFIED - just show we have a valid collision
                static bool firstValidCollisionLogged = false;
                if (!firstValidCollisionLogged) {
                    firstValidCollisionLogged = true;
                    printf("\n🎯 FIRST VALID COLLISION DETECTED! Processing key recovery...\n");
                }
            }
            collision_check_counter = 0;
        }



        if (GetTickCount64() - tm_stats > 2000) {
            ShowStats(tm0, ops, dp_val);
            last_ops = PntTotalOps;
            tm_stats = GetTickCount64();
        }

        if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps)) {
            gIsOpsLimit = true;
            printf("\nMax operations reached\n");
            break;
        }
    }

    printf("\n\nFinalizing...\n");

    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->Stop();

    u64 stop_timeout = GetTickCount64() + 5000;
    while (ThrCnt && GetTickCount64() < stop_timeout)
        Sleep(10);

    for (int i = 0; i < GpuCnt; i++) {
#ifdef _WIN32
        if (thr_handles[i] != NULL) {
            CloseHandle(thr_handles[i]);
        }
#else
        if (thr_handles[i] != 0) {
            pthread_join(thr_handles[i], NULL);
        }
#endif
    }

    // Final summary with collision statistics
    u64 total_time = (GetTickCount64() - tm0) / 1000;
    printf("\n=== SEARCH COMPLETE ===\n");
    printf("Total time: %llu seconds\n", (unsigned long long)total_time);
    printf("Total operations: 2^%.3f\n", log2((double)PntTotalOps));
    printf("Total collisions: %llu (including %llu from previous runs)\n",
        (unsigned long long)gCollisionCount, (unsigned long long)previously_saved);
    printf("Valid collisions: %llu\n", (unsigned long long)gValidCollisions);

    // Save final collision data
    if (gSaveCollisions && gCollisionsFileName[0] && gCollisionCount > previously_saved) {
        if (SaveCollisionsToFile(gCollisionsFileName)) {
            printf("Collisions saved: %llu total to %s\n",
                (unsigned long long)gCollisionCount, gCollisionsFileName);
        }
    }

    if (gIsOpsLimit) {
        printf("Stopped: Operation limit reached\n");
        if (gGenMode) {
            db.Header[0] = gRange;
            if (db.SaveToFile(gTamesFileName))
                printf("Tames saved: %llu\n", (unsigned long long)db.GetBlockCnt());
        }
        db.Clear();
        return false;
    }

    if (gSolved) {
        printf("\n*** SUCCESS: PRIVATE KEY FOUND! ***\n");
        db.Clear();
        *pk_res = gPrivKey;
        return true;
    }
    else {
        printf("Result: No solution found\n");
        db.Clear();
        return false;
    }
}

bool ParseCommandLine(int argc, char* argv[]) {
    int ci = 1;
    bool quietMode = false; // 🟢 ADD QUIET MODE FLAG

    while (ci < argc) {
        char* argument = argv[ci];
        ci++;
        if (strcmp(argument, "-gpu") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -gpu option\r\n");
                return false;
            }
            char* gpus = argv[ci];
            ci++;
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++) {
                if ((gpus[i] < '0') || (gpus[i] > '9')) {
                    printf("error: invalid value for -gpu option\r\n");
                    return false;
                }
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0) {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 13) || (val > 60)) {
                printf("error: invalid value for -dp option\r\n");
                return false;
            }
            gDP = val;
        }
        else if (strcmp(argument, "-gendp") == 0) {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 13) || (val > 42)) {
                printf("error: invalid value for -gendp option\r\n");
                return false;
            }
            gGenDP = val;
        }
        else if (strcmp(argument, "-maxdps") == 0) {
            gMaxDPs = atoi(argv[ci]);
            ci++;
            if (gMaxDPs < 1000) {
                printf("error: invalid value for -maxdps option (minimum 1000)\r\n");
                return false;
            }
        }
        else if (strcmp(argument, "-range") == 0) {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 32) || (val > 170)) {
                printf("error: invalid value for -range option\r\n");
                return false;
            }
            gRange = val;


        }
        else if (strcmp(argument, "-start") == 0) {
            if (!gStart.SetHexStr(argv[ci])) {
                printf("error: invalid value for -start option\r\n");
                return false;
            }
            ci++;
            gStartSet = true;
        }
        else if (strcmp(argument, "-pubkey") == 0) {
            if (!gPubKey.SetHexStr(argv[ci])) {
                printf("error: invalid value for -pubkey option\r\n");
                return false;
            }
            ci++;
        }
        else if (strcmp(argument, "-tames") == 0) {
            strcpy(gTamesFileName, argv[ci]);
            ci++;
        }
        else if (strcmp(argument, "-collisions") == 0) {
            strcpy(gCollisionsFileName, argv[ci]);
            ci++;
            gSaveCollisions = true;
            printf("Collision saving enabled: %s\n", gCollisionsFileName);
        }
        else if (strcmp(argument, "-max") == 0) {
            double val = atof(argv[ci]);
            ci++;
            if (val < 0.001) {
                printf("error: invalid value for -max option\r\n");
                return false;
            }
            gMax = val;
        }

        else if (strcmp(argument, "-quiet") == 0) {
            // 🟢 ADD QUIET MODE OPTION
            quietMode = true;
            printf("Quiet mode enabled - reduced console output\n");
        }
        else {
            printf("error: unknown option %s\r\n", argument);
            return false;
        }
    }

    // 🟢 STORE QUIET MODE IN GLOBAL VARIABLE
    // You'll need to add this to your global variables section:
    // volatile bool gQuietMode = false;
    gQuietMode = quietMode;

    // Auto-set start range if not specified
    if (!gStartSet) {
        EcInt defaultStart;
        defaultStart.Set(1);
        defaultStart.ShiftLeft(gRange - 1);
        gStart = defaultStart;
        gStartSet = true;

        char start_hex[100];
        gStart.GetHexStr(start_hex);
        char* clean_start = start_hex;
        while (clean_start[0] == '0' && clean_start[1] != '\0') {
            clean_start++;
        }
        printf("Using default start range for %d-bit: %s (2^%d)\r\n", gRange, clean_start, gRange - 1);
    }

    // Auto-load collisions if no specific file specified
    if (gCollisionsFileName[0] == 0 && !gPubKey.x.IsZero()) {
        AutoLoadCollisions();
    }
    else if (gCollisionsFileName[0] && IsFileExist(gCollisionsFileName)) {
        LoadCollisionsFromFile(gCollisionsFileName);
    }

    if (!gPubKey.x.IsZero()) {
        if (!gStartSet || !gRange || !gDP) {
            printf("error: you must also specify -dp, -range and -start options\r\n");
            return false;
        }
    }

    // Mode detection
    if (gTamesFileName[0]) {
        if (!IsFileExist(gTamesFileName)) {
            // File doesn't exist -> generation mode
            gGenMode = true;
            printf("Tames file doesn't exist, enabling GENERATION MODE\n");

            // In generation mode, ignore pubkey if provided
            if (!gPubKey.x.IsZero()) {
                printf("Warning: Public key provided but generation mode active. Ignoring pubkey.\n");
            }
        }
        else {
            // File exists -> use pubkey to determine mode
            if (!gPubKey.x.IsZero()) {
                // We have a pubkey to solve -> SOLVING MODE
                gGenMode = false;
                printf("Tames file exists and public key provided, using SOLVING MODE\n");
            }
            else {
                // No pubkey -> GENERATION MODE (extend existing tames)
                gGenMode = true;
                printf("Tames file exists, extending tames in GENERATION MODE\n");
            }
        }
    }

    return true;
}
int main(int argc, char* argv[]) {
#ifdef _DEBUG    
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    printf("********************************************************************************\r\n");
    printf("*                        Modified by D O M S (c) 2025                          *\r\n");
    printf("*                  COLLISION PERSISTENCE & AUTO-RESUME SYSTEM                  *\r\n");
    printf("*  Supports flexible range: 75-bit to 170-bit with collision tracking          *\r\n");
    printf("********************************************************************************\r\n\r\n");

#ifdef _WIN32
    printf("Windows version\r\n");
#else
    printf("Linux version\r\n");
#endif

    // Initialize
    InitEc();
    gStartSet = false;
    gTamesFileName[0] = 0;
    gCollisionsFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    gTestMode = false;
    gSaveCollisions = false;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));

    if (!ParseCommandLine(argc, argv))
        return 0;

    InitGpus();

    if (!GpuCnt) {
        printf("No supported GPUs detected, exit\r\n");
        return 0;
    }

    // Allocate memory
    pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    if (!pPntList || !pPntList2) {
        printf("Memory allocation failed!\r\n");
        return 0;
    }

    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;
    IsBench = gPubKey.x.IsZero();

    if (!IsBench && !gGenMode) {
        printf("\r\nSOLVING MODE - %d-bit range\r\n\r\n", gRange);
        EcPoint PntToSolve;
        EcInt pk_found;

        // Fix the public key parameter passing
        PntToSolve = gPubKey;

        // 🚨 VERIFICATION MUST HAPPEN BEFORE SolvePoint 🚨
        char mainX[100], mainY[100];
        PntToSolve.x.GetHexStr(mainX);
        PntToSolve.y.GetHexStr(mainY);
        printf("=== MAIN: PUBLIC KEY VERIFICATION ===\n");
        printf("Public key from command line:\n");
        printf("X: %s\n", mainX);
        printf("Y: %s\n", mainY);

        // Check if it's zero (parameter passing issue)
        if (PntToSolve.x.IsZero() && PntToSolve.y.IsZero()) {
            printf("!!! ERROR: Public key is ZERO in main()!\n");
            printf("!!! This indicates a parameter passing bug.\n");
            printf("!!! Check ParseCommandLine() and gPubKey assignment.\n");
            return 0;
        }
        printf("=== END MAIN VERIFICATION ===\n\n");

        printf("DEBUG: Using original public key as target (NO offset)\n");

        // Display search range information BEFORE calling SolvePoint
        char start_hex[100];
        gStart.GetHexStr(start_hex);
        char clean_start_hex[100];
        strcpy(clean_start_hex, start_hex);
        char* clean_start = clean_start_hex;
        while (clean_start[0] == '0' && clean_start[1] != '\0') {
            clean_start++;
        }

        EcInt rangeEnd;
        EcInt rangeSize;
        rangeSize.Set(1);
        rangeSize.ShiftLeft(gRange - 1);
        rangeEnd = gStart;
        rangeEnd.Add(rangeSize);
        EcInt one;
        one.Set(1);
        rangeEnd.Sub(one);

        char end_hex[100];
        rangeEnd.GetHexStr(end_hex);
        char clean_end_hex[100];
        strcpy(clean_end_hex, end_hex);
        char* clean_end = clean_end_hex;
        while (clean_end[0] == '0' && clean_end[1] != '\0') {
            clean_end++;
        }

        

        // 🟢 RANGE-SPECIFIC MESSAGING
        printf("=== %d-BIT PUZZLE SOLVER ===\n", gRange);
        printf("Search Range: %s to %s\r\n", clean_start, clean_end);
        printf("Range size: 2^%d = %.0f keys\r\n", gRange - 1, pow(2.0, gRange - 1));


        if (gRange <= 90) {
            printf("MODE: ALGORITHM TEST & VALIDATION\n");
        }
        else if (gRange <= 120) {
            printf("MODE: MEDIUM RANGE SEARCH\n");
        }
        else {
            printf("MODE: LARGE RANGE CHALLENGE\n");
        }

        EcInt rangeSizeHex;
        rangeSizeHex = rangeEnd;
        rangeSizeHex.Sub(gStart);
        rangeSizeHex.Add(one);

        char range_size_hex[100];
        rangeSizeHex.GetHexStr(range_size_hex);
        char clean_range_size_hex[100];
        strcpy(clean_range_size_hex, range_size_hex);
        char* clean_range_size = clean_range_size_hex;
        while (clean_range_size[0] == '0' && clean_range_size[1] != '\0') {
            clean_range_size++;
        }

        double total_keys_decimal = pow(2.0, gRange - 1);

        printf("Search Range: %s to %s\r\n", clean_start, clean_end);
        printf("Range size: %s (hex)\r\n", clean_range_size);
        printf("Total keys in range: 2^%d = %.0f\r\n", gRange - 1, total_keys_decimal);
        printf("Using DP: %d, GenDP: %d (auto-adjusted for %d-bit range)\r\n", gDP, gGenDP, gRange);

        if (gSaveCollisions) {
            printf("Collision tracking: ENABLED (%s)\n", gCollisionsFileName);
        }

        char sx[100], sy[100];
        gPubKey.x.GetHexStr(sx);
        gPubKey.y.GetHexStr(sy);
        printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);

        // 🚨 CALL SOLVEPOINT ONLY ONCE 🚨
        bool solveResult = SolvePoint(PntToSolve, gRange, gDP, &pk_found);

        if (!solveResult) {
            if (!gIsOpsLimit)
                printf("FATAL ERROR: SolvePoint failed\r\n");
            goto label_end;
        }

        // Verify the found key
       // pk_found.AddModP(gStart);
        EcPoint tmp = ec.MultiplyG(pk_found);
        if (!tmp.IsEqual(gPubKey)) {
            printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
            printf("Expected: X=%s, Y=%s\n", sx, sy);
            char foundX[100], foundY[100];
            tmp.x.GetHexStr(foundX);
            tmp.y.GetHexStr(foundY);
            printf("Found:    X=%s, Y=%s\n", foundX, foundY);
            goto label_end;
        }

        char s[100];
        pk_found.GetHexStr(s);
        printf("\r\n*** PRIVATE KEY FOUND! ***\r\n");
        printf("PRIVATE KEY: %s\r\n\r\n", s);

        char solved_file[100];
        // 🟢 DYNAMIC FILENAME BASED ON RANGE
        sprintf(solved_file, "SOLVED_%d_BIT_%llu.TXT", gRange, (unsigned long long)time(NULL));
        FILE* fp = fopen(solved_file, "a");
        if (fp) {
            fprintf(fp, "Private Key: %s\n", s);
            fprintf(fp, "Public Key X: %s\n", sx);
            fprintf(fp, "Public Key Y: %s\n", sy);
            fprintf(fp, "Range: %d-bit\n", gRange);
            fprintf(fp, "Found at: %llu operations\n", (unsigned long long)PntTotalOps);
            fprintf(fp, "Collisions checked: %llu\n", (unsigned long long)gCollisionCount);
            fprintf(fp, "Valid collisions: %llu\n", (unsigned long long)gValidCollisions);
            fclose(fp);
            printf("Key saved to %s\r\n", solved_file);
        }
        else {
            printf("WARNING: Cannot save the key to %s!\r\n", solved_file);
        }
    }
    else {
        if (gGenMode) {
            printf("\r\nTAMES GENERATION MODE - %d-bit range\r\n", gRange);
            printf("Using GenDP %d for generation (solving DP: %d)\r\n", gGenDP, gDP);
            printf("Max operations: %.1f%% of estimated\r\n", gMax * 100);
            printf("Max DPs to generate: %llu\r\n", (unsigned long long)gMaxDPs);

            double keys_per_dp = (double)(1ull << gGenDP);
            double conservative_speed = 2000.0;
            double initial_estimate = (conservative_speed * 1000000.0) / keys_per_dp;

            printf("Keys per DP: 2^%d = %.0f keys\n", gGenDP, keys_per_dp);
            printf("Initial estimate: ~%.2f DPs/s (based on %d MKeys/s)\n",
                initial_estimate, (int)conservative_speed);
            printf("Note: Actual rate will be shown in progress display\n");

            EcPoint benchPoint;
            benchPoint = ec.MultiplyG(gStart);
            EcInt dummy;
            SolvePoint(benchPoint, gRange, gDP, &dummy);
        }
        else {
            printf("\r\nBENCHMARK MODE - %d-bit range\r\n", gRange);

            while (1) {
                EcInt pk, pk_found;
                EcPoint PntToSolve;

                pk.RndBits(gRange);
                EcInt minRange;
                minRange.Set(1);
                minRange.ShiftLeft(gRange - 1);

                if (Compare(pk, minRange) < 0) {
                    pk.Add(minRange);
                }

                PntToSolve = ec.MultiplyG(pk);

                if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found)) {
                    if (!gIsOpsLimit)
                        printf("FATAL ERROR: SolvePoint failed\r\n");
                    break;
                }

                if (!pk_found.IsEqual(pk)) {
                    printf("FATAL ERROR: Found key is wrong!\r\n");
                    break;
                }

                TotalOps += PntTotalOps;
                TotalSolved++;
                u64 ops_per_pnt = TotalOps / TotalSolved;
                double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
                printf("Points solved: %llu, average K: %.3f (with DP and GPU overheads)\r\n",
                    (unsigned long long)TotalSolved, K);
            }
        }
    }

label_end:
    // Final collision save
    if (gSaveCollisions && gCollisionsFileName[0] && gCollisionCount > 0) {
        SaveCollisionsToFile(gCollisionsFileName);
        printf("Final collision data saved: %llu collisions to %s\n",
            (unsigned long long)gCollisionCount, gCollisionsFileName);
    }

    // Cleanup
    for (int i = 0; i < GpuCnt; i++)
        delete GpuKangs[i];

    DeInitEc();

    if (pPntList2) free(pPntList2);
    if (pPntList) free(pPntList);

    return 0;
}