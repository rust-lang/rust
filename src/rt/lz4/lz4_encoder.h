/*
   LZ4 Encoder - Part of LZ4 compression algorithm
   Copyright (C) 2011-2013, Yann Collet.
   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - LZ4 homepage : http://fastcompression.blogspot.com/p/lz4.html
   - LZ4 source repository : http://code.google.com/p/lz4/
*/

/* lz4_encoder.h must be included into lz4.c
   The objective of this file is to create a single LZ4 compression function source
   which will be instanciated multiple times with minor variations
   depending on a set of #define.
*/



//****************************
// Check required defines
//****************************

#ifndef FUNCTION_NAME
#  error "FUNTION_NAME is not defined"
#endif


//****************************
// Local definitions
//****************************

#ifdef COMPRESS_64K
#  define HASHLOG (MEMORY_USAGE-1)
#  define CURRENT_H_TYPE U16
#  define CURRENTBASE(base) const BYTE* const base = ip
#else
#  define HASHLOG (MEMORY_USAGE-2)
#  define CURRENT_H_TYPE HTYPE
#  define CURRENTBASE(base) INITBASE(base)
#endif

#define HASHTABLE_NBCELLS  (1U<<HASHLOG)
#define LZ4_HASH(i)        (((i) * 2654435761U) >> ((MINMATCH*8)-HASHLOG))
#define LZ4_HASHVALUE(p)   LZ4_HASH(A32(p))



//****************************
// Function code
//****************************

int FUNCTION_NAME(
#ifdef USE_HEAPMEMORY
                 void* ctx,
#endif
                 const char* source,
                 char* dest,
                 int inputSize
#ifdef LIMITED_OUTPUT
                ,int maxOutputSize
#endif
                 )
{
#ifdef USE_HEAPMEMORY
    CURRENT_H_TYPE* HashTable = (CURRENT_H_TYPE*)ctx;
#else
    CURRENT_H_TYPE HashTable[HASHTABLE_NBCELLS] = {0};
#endif

    const BYTE* ip = (BYTE*) source;
    CURRENTBASE(base);
    const BYTE* anchor = ip;
    const BYTE* const iend = ip + inputSize;
    const BYTE* const mflimit = iend - MFLIMIT;
#define matchlimit (iend - LASTLITERALS)

    BYTE* op = (BYTE*) dest;
#ifdef LIMITED_OUTPUT
    BYTE* const oend = op + maxOutputSize;
#endif

    int length;
    const int skipStrength = SKIPSTRENGTH;
    U32 forwardH;


    // Init
    if (inputSize<MINLENGTH) goto _last_literals;
#ifdef COMPRESS_64K
    if (inputSize>LZ4_64KLIMIT) return 0;   // Size too large (not within 64K limit)
#endif
#ifdef USE_HEAPMEMORY
    memset((void*)HashTable, 0, HASHTABLESIZE);
#endif

    // First Byte
    HashTable[LZ4_HASHVALUE(ip)] = (CURRENT_H_TYPE)(ip - base);
    ip++; forwardH = LZ4_HASHVALUE(ip);

    // Main Loop
    for ( ; ; )
    {
        int findMatchAttempts = (1U << skipStrength) + 3;
        const BYTE* forwardIp = ip;
        const BYTE* ref;
        BYTE* token;

        // Find a match
        do {
            U32 h = forwardH;
            int step = findMatchAttempts++ >> skipStrength;
            ip = forwardIp;
            forwardIp = ip + step;

            if unlikely(forwardIp > mflimit) { goto _last_literals; }

            forwardH = LZ4_HASHVALUE(forwardIp);
            ref = base + HashTable[h];
            HashTable[h] = (CURRENT_H_TYPE)(ip - base);

        } while ((ref < ip - MAX_DISTANCE) || (A32(ref) != A32(ip)));

        // Catch up
        while ((ip>anchor) && (ref>(BYTE*)source) && unlikely(ip[-1]==ref[-1])) { ip--; ref--; }

        // Encode Literal length
        length = (int)(ip - anchor);
        token = op++;
#ifdef LIMITED_OUTPUT
        if unlikely(op + length + (2 + 1 + LASTLITERALS) + (length>>8) > oend) return 0;   // Check output limit
#endif
        if (length>=(int)RUN_MASK) 
        { 
            int len = length-RUN_MASK; 
            *token=(RUN_MASK<<ML_BITS); 
            for(; len >= 255 ; len-=255) *op++ = 255; 
            *op++ = (BYTE)len; 
        }
        else *token = (BYTE)(length<<ML_BITS);

        // Copy Literals
        LZ4_BLINDCOPY(anchor, op, length);

_next_match:
        // Encode Offset
        LZ4_WRITE_LITTLEENDIAN_16(op,(U16)(ip-ref));

        // Start Counting
        ip+=MINMATCH; ref+=MINMATCH;    // MinMatch already verified
        anchor = ip;
        while likely(ip<matchlimit-(STEPSIZE-1))
        {
            UARCH diff = AARCH(ref) ^ AARCH(ip);
            if (!diff) { ip+=STEPSIZE; ref+=STEPSIZE; continue; }
            ip += LZ4_NbCommonBytes(diff);
            goto _endCount;
        }
        if (LZ4_ARCH64) if ((ip<(matchlimit-3)) && (A32(ref) == A32(ip))) { ip+=4; ref+=4; }
        if ((ip<(matchlimit-1)) && (A16(ref) == A16(ip))) { ip+=2; ref+=2; }
        if ((ip<matchlimit) && (*ref == *ip)) ip++;
_endCount:

        // Encode MatchLength
        length = (int)(ip - anchor);
#ifdef LIMITED_OUTPUT
        if unlikely(op + (1 + LASTLITERALS) + (length>>8) > oend) return 0;    // Check output limit
#endif
        if (length>=(int)ML_MASK) 
        { 
            *token += ML_MASK; 
            length -= ML_MASK; 
            for (; length > 509 ; length-=510) { *op++ = 255; *op++ = 255; } 
            if (length >= 255) { length-=255; *op++ = 255; } 
            *op++ = (BYTE)length; 
        }
        else *token += (BYTE)length;

        // Test end of chunk
        if (ip > mflimit) { anchor = ip;  break; }

        // Fill table
        HashTable[LZ4_HASHVALUE(ip-2)] = (CURRENT_H_TYPE)(ip - 2 - base);

        // Test next position
        ref = base + HashTable[LZ4_HASHVALUE(ip)];
        HashTable[LZ4_HASHVALUE(ip)] = (CURRENT_H_TYPE)(ip - base);
        if ((ref >= ip - MAX_DISTANCE) && (A32(ref) == A32(ip))) { token = op++; *token=0; goto _next_match; }

        // Prepare next loop
        anchor = ip++;
        forwardH = LZ4_HASHVALUE(ip);
    }

_last_literals:
    // Encode Last Literals
    {
        int lastRun = (int)(iend - anchor);
#ifdef LIMITED_OUTPUT
        if (((char*)op - dest) + lastRun + 1 + ((lastRun+255-RUN_MASK)/255) > (U32)maxOutputSize) return 0;  // Check output limit
#endif
        if (lastRun>=(int)RUN_MASK) { *op++=(RUN_MASK<<ML_BITS); lastRun-=RUN_MASK; for(; lastRun >= 255 ; lastRun-=255) *op++ = 255; *op++ = (BYTE) lastRun; }
        else *op++ = (BYTE)(lastRun<<ML_BITS);
        memcpy(op, anchor, iend - anchor);
        op += iend-anchor;
    }

    // End
    return (int) (((char*)op)-dest);
}



//****************************
// Clean defines
//****************************

// Required defines
#undef FUNCTION_NAME

// Locally Generated
#undef HASHLOG
#undef HASHTABLE_NBCELLS
#undef LZ4_HASH
#undef LZ4_HASHVALUE
#undef CURRENT_H_TYPE
#undef CURRENTBASE

// Optional defines
#ifdef LIMITED_OUTPUT
#undef LIMITED_OUTPUT
#endif

#ifdef USE_HEAPMEMORY
#undef USE_HEAPMEMORY
#endif
