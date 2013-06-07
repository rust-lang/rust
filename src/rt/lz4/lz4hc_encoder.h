/*
   LZ4 HC Encoder - Part of LZ4 HC algorithm
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

/* lz4hc_encoder.h must be included into lz4hc.c
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
#define COMBINED_NAME_RAW(n1,n2) n1 ## n2
#define COMBINED_NAME(n1,n2) COMBINED_NAME_RAW(n1,n2)
#define ENCODE_SEQUENCE_NAME COMBINED_NAME(FUNCTION_NAME,_encodeSequence)
#ifdef LIMITED_OUTPUT
#  define ENCODE_SEQUENCE(i,o,a,m,r,d) if (ENCODE_SEQUENCE_NAME(i,o,a,m,r,d)) return 0;
#else
#  define ENCODE_SEQUENCE(i,o,a,m,r,d) ENCODE_SEQUENCE_NAME(i,o,a,m,r)
#endif

//****************************
// Function code
//****************************

forceinline static int ENCODE_SEQUENCE_NAME (
                       const BYTE** ip, 
                       BYTE** op, 
                       const BYTE** anchor, 
                       int matchLength, 
                       const BYTE* ref
#ifdef LIMITED_OUTPUT
                      ,BYTE* oend
#endif
                       )
{
    int length, len; 
    BYTE* token;

    // Encode Literal length
    length = (int)(*ip - *anchor);
    token = (*op)++;
#ifdef LIMITED_OUTPUT
    if ((*op + length + (2 + 1 + LASTLITERALS) + (length>>8)) > oend) return 1; 		// Check output limit
#endif
    if (length>=(int)RUN_MASK) { *token=(RUN_MASK<<ML_BITS); len = length-RUN_MASK; for(; len > 254 ; len-=255) *(*op)++ = 255;  *(*op)++ = (BYTE)len; } 
    else *token = (BYTE)(length<<ML_BITS);

    // Copy Literals
    LZ4_BLINDCOPY(*anchor, *op, length);

    // Encode Offset
    LZ4_WRITE_LITTLEENDIAN_16(*op,(U16)(*ip-ref));

    // Encode MatchLength
    length = (int)(matchLength-MINMATCH);
#ifdef LIMITED_OUTPUT
    if (*op + (1 + LASTLITERALS) + (length>>8) > oend) return 1; 		// Check output limit
#endif
    if (length>=(int)ML_MASK) { *token+=ML_MASK; length-=ML_MASK; for(; length > 509 ; length-=510) { *(*op)++ = 255; *(*op)++ = 255; } if (length > 254) { length-=255; *(*op)++ = 255; } *(*op)++ = (BYTE)length; } 
    else *token += (BYTE)length;	

    // Prepare next loop
    *ip += matchLength;
    *anchor = *ip; 

    return 0;
}


int COMBINED_NAME(FUNCTION_NAME,_continue) (
                 void* ctxvoid,
                 const char* source, 
                 char* dest,
                 int inputSize
#ifdef LIMITED_OUTPUT
                ,int maxOutputSize
#endif
                )
{
    LZ4HC_Data_Structure* ctx = (LZ4HC_Data_Structure*) ctxvoid;
    const BYTE* ip = (const BYTE*) source;
    const BYTE* anchor = ip;
    const BYTE* const iend = ip + inputSize;
    const BYTE* const mflimit = iend - MFLIMIT;
    const BYTE* const matchlimit = (iend - LASTLITERALS);

    BYTE* op = (BYTE*) dest;
#ifdef LIMITED_OUTPUT
    BYTE* const oend = op + maxOutputSize;
#endif

    int	ml, ml2, ml3, ml0;
    const BYTE* ref=NULL;
    const BYTE* start2=NULL;
    const BYTE* ref2=NULL;
    const BYTE* start3=NULL;
    const BYTE* ref3=NULL;
    const BYTE* start0;
    const BYTE* ref0;

    // Ensure blocks follow each other
    if (ip != ctx->end) return 0;
    ctx->end += inputSize;

    ip++;

    // Main Loop
    while (ip < mflimit)
    {
        ml = LZ4HC_InsertAndFindBestMatch (ctx, ip, matchlimit, (&ref));
        if (!ml) { ip++; continue; }

        // saved, in case we would skip too much
        start0 = ip;
        ref0 = ref;
        ml0 = ml;

_Search2:
        if (ip+ml < mflimit)
            ml2 = LZ4HC_InsertAndGetWiderMatch(ctx, ip + ml - 2, ip + 1, matchlimit, ml, &ref2, &start2);
        else ml2 = ml;

        if (ml2 == ml)  // No better match
        {
            ENCODE_SEQUENCE(&ip, &op, &anchor, ml, ref, oend);
            continue;
        }

        if (start0 < ip)
        {
            if (start2 < ip + ml0)   // empirical
            {
                ip = start0;
                ref = ref0;
                ml = ml0;
            }
        }

        // Here, start0==ip
        if ((start2 - ip) < 3)   // First Match too small : removed
        {
            ml = ml2;
            ip = start2;
            ref =ref2;
            goto _Search2;
        }

_Search3:
        // Currently we have :
        // ml2 > ml1, and
        // ip1+3 <= ip2 (usually < ip1+ml1)
        if ((start2 - ip) < OPTIMAL_ML)
        {
            int correction;
            int new_ml = ml;
            if (new_ml > OPTIMAL_ML) new_ml = OPTIMAL_ML;
            if (ip+new_ml > start2 + ml2 - MINMATCH) new_ml = (int)(start2 - ip) + ml2 - MINMATCH;
            correction = new_ml - (int)(start2 - ip);
            if (correction > 0)
            {
                start2 += correction;
                ref2 += correction;
                ml2 -= correction;
            }
        }
        // Now, we have start2 = ip+new_ml, with new_ml = min(ml, OPTIMAL_ML=18)

        if (start2 + ml2 < mflimit)
            ml3 = LZ4HC_InsertAndGetWiderMatch(ctx, start2 + ml2 - 3, start2, matchlimit, ml2, &ref3, &start3);
        else ml3 = ml2;

        if (ml3 == ml2) // No better match : 2 sequences to encode
        {
            // ip & ref are known; Now for ml
            if (start2 < ip+ml)  ml = (int)(start2 - ip);
            // Now, encode 2 sequences
            ENCODE_SEQUENCE(&ip, &op, &anchor, ml, ref, oend);
            ip = start2;
            ENCODE_SEQUENCE(&ip, &op, &anchor, ml2, ref2, oend);
            continue;
        }

        if (start3 < ip+ml+3) // Not enough space for match 2 : remove it
        {
            if (start3 >= (ip+ml)) // can write Seq1 immediately ==> Seq2 is removed, so Seq3 becomes Seq1
            {
                if (start2 < ip+ml)
                {
                    int correction = (int)(ip+ml - start2);
                    start2 += correction;
                    ref2 += correction;
                    ml2 -= correction;
                    if (ml2 < MINMATCH)
                    {
                        start2 = start3;
                        ref2 = ref3;
                        ml2 = ml3;
                    }
                }

                ENCODE_SEQUENCE(&ip, &op, &anchor, ml, ref, oend);
                ip  = start3;
                ref = ref3;
                ml  = ml3;

                start0 = start2;
                ref0 = ref2;
                ml0 = ml2;
                goto _Search2;
            }

            start2 = start3;
            ref2 = ref3;
            ml2 = ml3;
            goto _Search3;
        }

        // OK, now we have 3 ascending matches; let's write at least the first one
        // ip & ref are known; Now for ml
        if (start2 < ip+ml)
        {
            if ((start2 - ip) < (int)ML_MASK)
            {
                int correction;
                if (ml > OPTIMAL_ML) ml = OPTIMAL_ML;
                if (ip + ml > start2 + ml2 - MINMATCH) ml = (int)(start2 - ip) + ml2 - MINMATCH;
                correction = ml - (int)(start2 - ip);
                if (correction > 0)
                {
                    start2 += correction;
                    ref2 += correction;
                    ml2 -= correction;
                }
            }
            else
            {
                ml = (int)(start2 - ip);
            }
        }
        ENCODE_SEQUENCE(&ip, &op, &anchor, ml, ref, oend);

        ip = start2;
        ref = ref2;
        ml = ml2;

        start2 = start3;
        ref2 = ref3;
        ml2 = ml3;

        goto _Search3;

    }

    // Encode Last Literals
    {
        int lastRun = (int)(iend - anchor);
#ifdef LIMITED_OUTPUT
        if (((char*)op - dest) + lastRun + 1 + ((lastRun+255-RUN_MASK)/255) > (U32)maxOutputSize) return 0;  // Check output limit
#endif
        if (lastRun>=(int)RUN_MASK) { *op++=(RUN_MASK<<ML_BITS); lastRun-=RUN_MASK; for(; lastRun > 254 ; lastRun-=255) *op++ = 255; *op++ = (BYTE) lastRun; } 
        else *op++ = (BYTE)(lastRun<<ML_BITS);
        memcpy(op, anchor, iend - anchor);
        op += iend-anchor;
    } 

    // End
    return (int) (((char*)op)-dest);
}


int FUNCTION_NAME (const char* source, 
                 char* dest,
                 int inputSize
#ifdef LIMITED_OUTPUT
                ,int maxOutputSize
#endif
                )
{
    void* ctx = LZ4_createHC(source);
    int result;
    if (ctx==NULL) return 0;
#ifdef LIMITED_OUTPUT
    result = COMBINED_NAME(FUNCTION_NAME,_continue) (ctx, source, dest, inputSize, maxOutputSize);
#else
    result = COMBINED_NAME(FUNCTION_NAME,_continue) (ctx, source, dest, inputSize);
#endif
    LZ4_freeHC(ctx);

    return result;
}


//****************************
// Clean defines
//****************************

// Required defines
#undef FUNCTION_NAME

// Locally Generated
#undef ENCODE_SEQUENCE
#undef ENCODE_SEQUENCE_NAME

// Optional defines
#ifdef LIMITED_OUTPUT
#undef LIMITED_OUTPUT
#endif


