// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include "Ec.h"

#define STATS_WND_SIZE	16

struct EcJMP
{
    EcPoint p;
    EcInt dist;
};

//96bytes size
struct TPointPriv
{
    u64 x[4];
    u64 y[4];
    u64 priv[4];
};

extern EcInt gStart;

class RCGpuKang
{
private:
    bool StopFlag = false;
    EcPoint PntToSolve;
    int Range = 0; //in bits
    int DP = 0; //in bits
    Ec ec;
    EcInt StartRange;  // ?? ADD THIS LINE - to store the start range
    int GetHighestBit(const EcInt& value);
    bool IsInTargetRange(const EcInt& value, int range); // <-- Line in header
    void DebugRangeVerification(int range);
    EcInt Int_Start;   // <-- ADD THIS AT THE TOP OF THE FILE

    u32* DPs_out = nullptr;
    TKparams Kparams;

    EcInt HalfRange;
    EcPoint PntHalfRange;
    EcPoint NegPntHalfRange;
    TPointPriv* RndPnts = nullptr;
    EcJMP* EcJumps1 = nullptr;
    EcJMP* EcJumps2 = nullptr;
    EcJMP* EcJumps3 = nullptr;

    EcPoint PntA;
    EcPoint PntB;

    int cur_stats_ind = 0;
    int SpeedStats[STATS_WND_SIZE] = { 0 };

    void GenerateRndDistances();
    bool Start();
    void Release();
#ifdef DEBUG_MODE
    int Dbg_CheckKangs();
#endif
public:
    int persistingL2CacheMaxSize = 0;
    int CudaIndex = -1; //gpu index in cuda
    int mpCnt = 0;
    int KangCnt = 0;
    bool Failed = false;
    bool IsOldGpu = false;
    size_t allocatedMemoryMB = 0; // Track allocated GPU memory in MB

    int CalcKangCnt();
    bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
    void Stop();
    void Execute();

    u32 dbg[256] = { 0 };

    int GetStatsSpeed();
};