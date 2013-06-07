/*
   LZ4 - Fast LZ compression algorithm
   Header File
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
#pragma once

#if defined (__cplusplus)
extern "C" {
#endif


//**************************************
// Compiler Options
//**************************************
#if defined(_MSC_VER) && !defined(__cplusplus)   // Visual Studio
#  define inline __inline           // Visual is not C99, but supports some kind of inline
#endif


//****************************
// Simple Functions
//****************************

int LZ4_compress        (const char* source, char* dest, int inputSize);
int LZ4_decompress_safe (const char* source, char* dest, int inputSize, int maxOutputSize);

/*
LZ4_compress() :
    Compresses 'inputSize' bytes from 'source' into 'dest'.
    Destination buffer must be already allocated,
    and must be sized to handle worst cases situations (input data not compressible)
    Worst case size evaluation is provided by function LZ4_compressBound()
    inputSize : Max supported value is ~1.9GB
    return : the number of bytes written in buffer dest
             or 0 if the compression fails

LZ4_decompress_safe() :
    maxOutputSize : is the size of the destination buffer (which must be already allocated)
    return : the number of bytes decoded in the destination buffer (necessarily <= maxOutputSize)
             If the source stream is malformed, the function will stop decoding and return a negative result.
             This function never writes outside of output buffer, and never reads outside of input buffer. It is therefore protected against malicious data packets
*/


//****************************
// Advanced Functions
//****************************

static inline int LZ4_compressBound(int isize)   { return ((isize) + ((isize)/255) + 16); }
#define           LZ4_COMPRESSBOUND(    isize)            ((isize) + ((isize)/255) + 16)

/*
LZ4_compressBound() :
    Provides the maximum size that LZ4 may output in a "worst case" scenario (input data not compressible)
    primarily useful for memory allocation of output buffer.
	inline function is recommended for the general case,
	macro is also provided when result needs to be evaluated at compile time (such as table size allocation).

    isize  : is the input size. Max supported value is ~1.9GB
    return : maximum output size in a "worst case" scenario
    note : this function is limited by "int" range (2^31-1)
*/


int LZ4_compress_limitedOutput (const char* source, char* dest, int inputSize, int maxOutputSize);

/*
LZ4_compress_limitedOutput() :
    Compress 'inputSize' bytes from 'source' into an output buffer 'dest' of maximum size 'maxOutputSize'.
    If it cannot achieve it, compression will stop, and result of the function will be zero.
    This function never writes outside of provided output buffer.

    inputSize  : Max supported value is ~1.9GB
    maxOutputSize : is the size of the destination buffer (which must be already allocated)
    return : the number of bytes written in buffer 'dest'
             or 0 if the compression fails
*/


int LZ4_decompress_fast (const char* source, char* dest, int outputSize);

/*
LZ4_decompress_fast() :
    outputSize : is the original (uncompressed) size
    return : the number of bytes read from the source buffer (in other words, the compressed size)
             If the source stream is malformed, the function will stop decoding and return a negative result.
    note : This function is a bit faster than LZ4_decompress_safe()
           This function never writes outside of output buffers, and never read before input buffer, but may read beyond input buffer (since it doesn't know its size) in case of malicious data packet.
           Use this function preferably into a trusted environment (data to decode comes from a trusted source).
           Destination buffer must be already allocated. Its size must be a minimum of 'outputSize' bytes.
*/

int LZ4_decompress_safe_partial (const char* source, char* dest, int inputSize, int targetOutputSize, int maxOutputSize);

/*
LZ4_decompress_safe_partial() :
    This function decompress a compressed block of size 'inputSize' at position 'source'
    into output buffer 'dest' of size 'maxOutputSize'.
    The function stops decompressing operation as soon as 'targetOutputSize' has been reached,
    reducing decompression time.
    return : the number of bytes decoded in the destination buffer (necessarily <= maxOutputSize)
       Note : this number might be < 'targetOutputSize' if the number of bytes to decode into the compressed block is not enough.
             Always control how many bytes were  decoded.
             If the source stream is malformed, the function will stop decoding and return a negative result.
             This function never writes outside of output buffer, and never reads outside of input buffer. It is therefore protected against malicious data packets
*/


int LZ4_decompress_safe_withPrefix64k (const char* source, char* dest, int inputSize, int maxOutputSize);
int LZ4_decompress_fast_withPrefix64k (const char* source, char* dest, int outputSize);

/*
*_withPrefix64k() :
    These decoding functions work the same as their "normal name" versions,
    but will potentially use up to 64KB of data in front of 'char* dest'.
    These functions are used for decoding inter-dependant blocks.
*/


//****************************
// Obsolete Functions
//****************************

static inline int LZ4_uncompress (const char* source, char* dest, int outputSize)   { return LZ4_decompress_fast(source, dest, outputSize); }
static inline int LZ4_uncompress_unknownOutputSize (const char* source, char* dest, int isize, int maxOutputSize)   { return LZ4_decompress_safe(source, dest, isize, maxOutputSize); }

/* 
These functions are deprecated and should no longer be used.
They are provided here for compatibility with existing user programs.
*/



#if defined (__cplusplus)
}
#endif
