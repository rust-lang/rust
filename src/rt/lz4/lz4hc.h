/*
   LZ4 HC - High Compression Mode of LZ4
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


int LZ4_compressHC (const char* source, char* dest, int inputSize);
/*
LZ4_compressHC :
    return : the number of bytes in compressed buffer dest
             or 0 if compression fails.
    note : destination buffer must be already allocated. 
        To avoid any problem, size it to handle worst cases situations (input data not compressible)
        Worst case size evaluation is provided by function LZ4_compressBound() (see "lz4.h")
*/

int LZ4_compressHC_limitedOutput (const char* source, char* dest, int inputSize, int maxOutputSize);
/*
LZ4_compress_limitedOutput() :
    Compress 'inputSize' bytes from 'source' into an output buffer 'dest' of maximum size 'maxOutputSize'.
    If it cannot achieve it, compression will stop, and result of the function will be zero.
    This function never writes outside of provided output buffer.

    inputSize  : Max supported value is 1 GB
    maxOutputSize : is maximum allowed size into the destination buffer (which must be already allocated)
    return : the number of output bytes written in buffer 'dest'
             or 0 if compression fails.
*/


/* Note :
Decompression functions are provided within LZ4 source code (see "lz4.h") (BSD license)
*/


/* Advanced Functions
*/

void* LZ4_createHC (const char* slidingInputBuffer);
int   LZ4_compressHC_continue (void* LZ4HC_Data, const char* source, char* dest, int inputSize);
int   LZ4_compressHC_limitedOutput_continue (void* LZ4HC_Data, const char* source, char* dest, int inputSize, int maxOutputSize);
char* LZ4_slideInputBufferHC (void* LZ4HC_Data);
int   LZ4_freeHC (void* LZ4HC_Data);

/* 
These functions allow the compression of dependent blocks, where each block benefits from prior 64 KB within preceding blocks.
In order to achieve this, it is necessary to start creating the LZ4HC Data Structure, thanks to the function :

void* LZ4_createHC (const char* slidingInputBuffer);
The result of the function is the (void*) pointer on the LZ4HC Data Structure.
This pointer will be needed in all other functions.
If the pointer returned is NULL, then the allocation has failed, and compression must be aborted.
The only parameter 'const char* slidingInputBuffer' must, obviously, point at the beginning of input buffer.
The input buffer must be already allocated, and size at least 192KB.
'slidingInputBuffer' will also be the 'const char* source' of the first block.

All blocks are expected to lay next to each other within the input buffer, starting from 'slidingInputBuffer'.
To compress each block, use either LZ4_compressHC_continue() or LZ4_compressHC_limitedOutput_continue().
Their behavior are identical to LZ4_compressHC() or LZ4_compressHC_limitedOutput(), 
but require the LZ4HC Data Structure as their first argument, and check that each block starts right after the previous one.
If next block does not begin immediately after the previous one, the compression will fail (return 0).

When it's no longer possible to lay the next block after the previous one (not enough space left into input buffer), a call to : 
char* LZ4_slideInputBufferHC(void* LZ4HC_Data);
must be performed. It will typically copy the latest 64KB of input at the beginning of input buffer.
Note that, for this function to work properly, minimum size of an input buffer must be 192KB.
==> The memory position where the next input data block must start is provided as the result of the function.

Compression can then resume, using LZ4_compressHC_continue() or LZ4_compressHC_limitedOutput_continue(), as usual.

When compression is completed, a call to LZ4_freeHC() will release the memory used by the LZ4HC Data Structure.
*/


#if defined (__cplusplus)
}
#endif
