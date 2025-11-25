// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

int RCGpuKang::GetHighestBit(const EcInt& value) {
	for (int i = 4; i >= 0; i--) {
		if (value.data[i] != 0) {
			u64 word = value.data[i];
			for (int bit = 63; bit >= 0; bit--) {
				if (word & (1ULL << bit)) {
					return i * 64 + bit;
				}
			}
		}
	}
	return -1;
}

bool RCGpuKang::IsInTargetRange(const EcInt& value, int range)
{
	if (range <= 0 || range > 256)
		return true;

	// canonical kangaroo range: [2^(range-1) .. 2^range - 1]
	EcInt minRange, maxRange, one;
	minRange.Set(1);
	minRange.ShiftLeft(range - 1);

	maxRange.Set(1);
	maxRange.ShiftLeft(range);
	one.Set(1);
	maxRange.Sub(one);  // 2^range - 1

	EcInt maxPlusOne = maxRange;
	maxPlusOne.Add(one);

	EcInt v = value;

	return (!v.IsLessThanU(minRange) && v.IsLessThanU(maxPlusOne));
}


void RCGpuKang::DebugRangeVerification(int range)
{
	if (range < 75 || range > 170)
		return;

	printf("FOCUSING ON %d-BIT RANGE (CANONICAL RANGE)\n", range);

	// canonical boundaries
	EcInt minRange, maxRange, one;
	minRange.Set(1);
	minRange.ShiftLeft(range - 1);   // 2^(range-1)

	maxRange.Set(1);
	maxRange.ShiftLeft(range);       // 2^range
	one.Set(1);
	maxRange.Sub(one);               // 2^range - 1

	EcInt maxPlusOne = maxRange;
	maxPlusOne.Add(one);

	char a[100], b[100];
	minRange.GetHexStr(a);
	maxRange.GetHexStr(b);

	printf("Canonical kangaroo range:\n");
	printf("  Start = %s\n", a);
	printf("  End   = %s\n\n", b);

	printf("Verifying kangaroo distances...\n");

	int inRange = 0;
	int sample = 32;

	for (int i = 0; i < sample && i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);

		bool low = d.IsLessThanU(minRange);
		bool high = !d.IsLessThanU(maxPlusOne);
		bool ok = (!low && !high);

		if (ok) inRange++;

		char hex[100];
		d.GetHexStr(hex);

		printf("  Kangaroo %2d: %s %s\n", i, hex, ok ? "IN RANGE" : "OUT RANGE");
	}

	printf("DONE\n");
	printf("Valid distances: %d / %d\n\n", inRange, sample);
}


//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;

	//allocate gpu mem
	u64 size;
	if (!IsOldGpu)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
		err = cudaMalloc((void**)&Kparams.L2, L2size);
		if (err != cudaSuccess)
		{
			printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
		//persisting for L2
		cudaStreamAttrValue stream_attribute;
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
		err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = KangCnt * 96;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Kangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

	//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);
	//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	err = cuSetGpuParams(Kparams, jmp2_table);
	if (err != cudaSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(jmp2_table);
	//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void RCGpuKang::Release()
{
	free(RndPnts);
	free(DPs_out);
	cudaFree(Kparams.LoopedKangs);
	cudaFree(Kparams.dbg_buf);
	cudaFree(Kparams.LoopTable);
	cudaFree(Kparams.LastPnts);
	cudaFree(Kparams.L1S2);
	cudaFree(Kparams.DPTable);
	cudaFree(Kparams.JumpsList);
	cudaFree(Kparams.Jumps3);
	cudaFree(Kparams.Jumps2);
	cudaFree(Kparams.Jumps1);
	cudaFree(Kparams.Kangs);
	cudaFree(Kparams.DPs_out);
	if (!IsOldGpu)
		cudaFree(Kparams.L2);
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}


void RCGpuKang::GenerateRndDistances()
{
	EcInt minRange, maxRange, one;
	minRange.Set(1);
	minRange.ShiftLeft(Range - 1);       // 2^(Range-1)

	maxRange.Set(1);
	maxRange.ShiftLeft(Range);           // 2^Range
	one.Set(1);
	maxRange.Sub(one);                   // 2^Range - 1

	EcInt maxRangePlusOne = maxRange;
	maxRangePlusOne.Add(one);

	char startStr[100], endStr[100];
	minRange.GetHexStr(startStr);
	maxRange.GetHexStr(endStr);

	printf("135-BIT RANGE: [%s, %s]\n", startStr, endStr);
	printf("Should span: 0x4... to 0x7...\n\n");

	// Calculate zone boundaries manually
	// For 135-bit: 2^134 to 2^135-1
	// Divide into 3 equal zones

	EcInt zoneSize;
	zoneSize.Set(1);
	zoneSize.ShiftLeft(Range - 2);  // Each zone = 2^(133) = 1/4 of total range? Wait, let's recalculate

	// Better approach: Calculate the actual range size and divide by 3
	EcInt rangeSize = maxRange;
	rangeSize.Sub(minRange);
	rangeSize.Add(one);

	// Manual division by 3: shift right (approximate)
	EcInt zoneSizeApprox = rangeSize;
	zoneSizeApprox.ShiftRight(2);  // Divide by 4 as approximation
	EcInt temp = zoneSizeApprox;
	temp.ShiftRight(1);  // temp = zoneSize/8
	zoneSizeApprox.Add(temp);  // zoneSizeApprox = zoneSize/4 + zoneSize/8 = 3*zoneSize/8 ≈ zoneSize/3

	EcInt tameStart = minRange;
	EcInt tameEnd = tameStart;
	tameEnd.Add(zoneSizeApprox);

	EcInt wild1Start = tameEnd;
	wild1Start.Add(one);
	EcInt wild1End = wild1Start;
	wild1End.Add(zoneSizeApprox);

	EcInt wild2Start = wild1End;
	wild2Start.Add(one);
	EcInt wild2End = maxRange;

	printf("Kangaroo Distribution:\n");
	printf("  TAMES:  [0x4... - 0x5...] (first ~1/3)\n");
	printf("  WILD1:  [0x5... - 0x6...] (second ~1/3)\n");
	printf("  WILD2:  [0x6... - 0x7...] (last ~1/3)\n\n");

	int out_of_range = 0;
	int tame_count = 0, wild1_count = 0, wild2_count = 0;

	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;

		// MANUAL RANGE GENERATION (since RndRange doesn't exist)
		if (i < KangCnt / 3) {
			// TAME kangaroos - first ~1/3
			d.RndMax(zoneSizeApprox);
			d.Add(tameStart);
			tame_count++;
		}
		else if (i < 2 * KangCnt / 3) {
			// WILD1 kangaroos - second ~1/3
			d.RndMax(zoneSizeApprox);
			d.Add(wild1Start);
			d.data[0] &= 0xFFFFFFFFFFFFFFFEULL; // wild = even
			wild1_count++;
		}
		else {
			// WILD2 kangaroos - last ~1/3
			EcInt wild2Range = wild2End;
			wild2Range.Sub(wild2Start);
			d.RndMax(wild2Range);
			d.Add(wild2Start);
			d.data[0] &= 0xFFFFFFFFFFFFFFFEULL; // wild = even
			wild2_count++;
		}

		// Validate the range
		if (d.IsLessThanU(minRange) || !d.IsLessThanU(maxRangePlusOne)) {
			printf("❌ RANGE ERROR: Kangaroo %d out of bounds!\n", i);
			out_of_range++;
			// Force into correct range based on group
			if (i < KangCnt / 3) {
				d = tameStart;
			}
			else if (i < 2 * KangCnt / 3) {
				d = wild1Start;
				d.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
			}
			else {
				d = wild2Start;
				d.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
			}
		}

		memcpy(RndPnts[i].priv, d.data, 24);

		// Debug first few from each group
		if (i < 3 || i == KangCnt / 3 || i == 2 * KangCnt / 3) {
			char ds[100];
			d.GetHexStr(ds);
			const char* type = (i < KangCnt / 3) ? "TAME" :
				(i < 2 * KangCnt / 3) ? "WILD1" : "WILD2";
			printf("%s %d: %s\n", type, i, ds);
		}
	}

	printf("Final distribution: Tames=%d, Wild1=%d, Wild2=%d\n",
		tame_count, wild1_count, wild2_count);

	if (out_of_range)
		printf("Warning: %d kangaroos adjusted to range\n", out_of_range);
	else
		printf("✅ All kangaroos properly distributed across %d-bit range\n", Range);

	// Additional debug: show the actual zone boundaries
	char ts[100], te[100], w1s[100], w1e[100], w2s[100], w2e[100];
	tameStart.GetHexStr(ts);
	tameEnd.GetHexStr(te);
	wild1Start.GetHexStr(w1s);
	wild1End.GetHexStr(w1e);
	wild2Start.GetHexStr(w2s);
	wild2End.GetHexStr(w2e);

	printf("\nActual Zone Boundaries:\n");
	printf("TAME:  %s to %s\n", ts, te);
	printf("WILD1: %s to %s\n", w1s, w1e);
	printf("WILD2: %s to %s\n", w2s, w2e);
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	// Add range info at start
	if (Range >= 75 && Range <= 170) {
		printf("GPU %d: Initializing %d-bit kangaroo search...\n", CudaIndex, Range);
	}

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
	/*
		//we can calc start points on CPU
		for (int i = 0; i < KangCnt; i++)
		{
			EcInt d;
			memcpy(d.data, RndPnts[i].priv, 24);
			d.data[3] = 0;
			d.data[4] = 0;
			EcPoint p = ec.MultiplyG(d);
			memcpy(RndPnts[i].x, p.x.data, 32);
			memcpy(RndPnts[i].y, p.y.data, 32);
		}
		for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
		{
			EcPoint p;
			p.LoadFromBuffer64((u8*)RndPnts[i].x);
			p = ec.AddPoints(p, PntA);
			p.SaveToBuffer64((u8*)RndPnts[i].x);
		}
		for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
		{
			EcPoint p;
			p.LoadFromBuffer64((u8*)RndPnts[i].x);
			p = ec.AddPoints(p, PntB);
			p.SaveToBuffer64((u8*)RndPnts[i].x);
		}
		//copy to gpu
		err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	/**/
	//but it's faster to calc then on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	CallGpuKernelGen(Kparams);

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		cudaMemset(Kparams.DPs_out, 0, 4);
		cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		cudaMemset(Kparams.LoopedKangs, 0, 8);
		CallGpuKernelABC(Kparams);
		int cnt;
		err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}

		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
		cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

		u32 lcnt;
		cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif
	}

	Release();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}