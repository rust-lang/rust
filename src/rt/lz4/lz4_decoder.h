/*
   LZ4 Decoder - Part of LZ4 compression algorithm
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

/* lz4_decoder.h must be included into lz4.c
   The objective of this file is to create a single LZ4 decoder function source
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
// Control tests
//****************************

#ifdef EXITCONDITION_INPUTSIZE
#  define INPUTBUFFER_CONTROL(ip,iend) likely(ip<iend)
#else
#  define INPUTBUFFER_CONTROL(ip,iend) (1)
#endif

#ifdef PARTIAL_DECODING
#  define OUTPUTTARGET(cpy,oexit) (cpy >= oexit)
#else
#  define OUTPUTTARGET(cpy,oexit) (0)
#endif




//****************************
// Function code
//****************************

int FUNCTION_NAME(const char* source,
                 char* dest,
#ifdef EXITCONDITION_INPUTSIZE
                 int inputSize,
#endif
#ifdef PARTIAL_DECODING
                 int targetOutputSize,
#endif
                 int outputSize
                 )
{
    // Local Variables
    const BYTE* restrict ip = (const BYTE*) source;
    const BYTE* ref;
#ifdef EXITCONDITION_INPUTSIZE
    const BYTE* const iend = ip + inputSize;
#endif

    BYTE* op = (BYTE*) dest;
    BYTE* const oend = op + outputSize;
    BYTE* cpy;
#ifdef PARTIAL_DECODING
    BYTE* const oexit = op + targetOutputSize;
#endif

    size_t dec32table[] = {0, 3, 2, 3, 0, 0, 0, 0};
#if LZ4_ARCH64
    size_t dec64table[] = {0, 0, 0, (size_t)-1, 0, 1, 2, 3};
#endif


#ifdef EXITCONDITION_INPUTSIZE
    // Special case
    if unlikely(!inputSize) goto _output_error;     // A correctly formed null-compressed LZ4 must have at least one byte (token=0)
#endif

    // Main Loop
    while (1)
    {
        unsigned token;
        size_t length;

        // get runlength
        token = *ip++;
        if ((length=(token>>ML_BITS)) == RUN_MASK)  
        { 
            unsigned s=255; 
            while (INPUTBUFFER_CONTROL(ip,iend) && (s==255)) 
            { 
                s=*ip++; 
                length += s; 
            } 
        }

        // copy literals
        cpy = op+length;
#ifdef EXITCONDITION_INPUTSIZE
        if ((cpy>oend-MFLIMIT) || (ip+length>iend-(2+1+LASTLITERALS)) || OUTPUTTARGET(cpy,oexit))
        {
            if (cpy > oend) goto _output_error;          // Error : write attempt beyond end of output buffer
            if ((!OUTPUTTARGET(cpy,oexit)) && (ip+length != iend)) goto _output_error;   // Error : Must consume all input at this stage, except if reaching TargetOutputSize
#else
        if (cpy>oend-COPYLENGTH)
        {
            if (cpy != oend) goto _output_error;         // Error : not enough place for another match (min 4) + 5 literals
#endif
            memcpy(op, ip, length);
            ip += length;
            op += length;
            break;                                       // Necessarily EOF, due to parsing restrictions
        }
        LZ4_WILDCOPY(ip, op, cpy); ip -= (op-cpy); op = cpy;

        // get offset
        LZ4_READ_LITTLEENDIAN_16(ref,cpy,ip); ip+=2;
#ifndef PREFIX_64K
        if unlikely(ref < (BYTE* const)dest) goto _output_error;   // Error : offset outside destination buffer
#endif

        // get matchlength
        if ((length=(token&ML_MASK)) == ML_MASK) 
        { 
            while INPUTBUFFER_CONTROL(ip,iend-(LASTLITERALS+1))    // A minimum nb of input bytes must remain for LASTLITERALS + token
            { 
                unsigned s = *ip++; 
                length += s; 
                if (s==255) continue; 
                break; 
            } 
        }

        // copy repeated sequence
        if unlikely((op-ref)<STEPSIZE)
        {
#if LZ4_ARCH64
            size_t dec64 = dec64table[op-ref];
#else
            const size_t dec64 = 0;
#endif
            op[0] = ref[0];
            op[1] = ref[1];
            op[2] = ref[2];
            op[3] = ref[3];
            op += 4, ref += 4; ref -= dec32table[op-ref];
            A32(op) = A32(ref); 
            op += STEPSIZE-4; ref -= dec64;
        } else { LZ4_COPYSTEP(ref,op); }
        cpy = op + length - (STEPSIZE-4);

        if unlikely(cpy>oend-(COPYLENGTH)-(STEPSIZE-4))
        {
            if (cpy > oend-LASTLITERALS) goto _output_error;    // Error : last 5 bytes must be literals
            LZ4_SECURECOPY(ref, op, (oend-COPYLENGTH));
            while(op<cpy) *op++=*ref++;
            op=cpy;
            continue;
        }
        
        LZ4_WILDCOPY(ref, op, cpy);
        op=cpy;		// correction
    }

    // end of decoding
#ifdef EXITCONDITION_INPUTSIZE
    return (int) (((char*)op)-dest);     // Nb of output bytes decoded
#else
    return (int) (((char*)ip)-source);   // Nb of input bytes read
#endif

    // Overflow error detected
_output_error:
    return (int) (-(((char*)ip)-source))-1;
}



//****************************
// Clean defines
//****************************

// Required defines
#undef FUNCTION_NAME

// Locally Generated
#undef INPUTBUFFER_CONTROL
#undef OUTPUTTARGET

// Optional defines
#ifdef EXITCONDITION_INPUTSIZE
#undef EXITCONDITION_INPUTSIZE
#endif

#ifdef PREFIX_64K
#undef PREFIX_64K
#endif

#ifdef PARTIAL_DECODING
#undef PARTIAL_DECODING
#endif

