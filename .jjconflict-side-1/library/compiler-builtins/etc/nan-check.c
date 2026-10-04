/* Print the representation of qNaNs, sNaNs, and quieted sNaNa.

The behavior of NaN can be surprising on targets that don't use the typical
quiet bit, as defined IEEE 754-2008. Most targets print the following:

    bf_nan()                           0x7fc0
    bf_snan()                          0x7fa0
    quietb16(bf_nan())                 0x7fc0
    quietb16(bf_snan())                0x7fe0

    __builtin_nanf16("")               0x7e00
    __builtin_nansf16("")              0x7d00
    quiet16(__builtin_nanf16(""))      0x7e00
    quiet16(__builtin_nansf16(""))     0x7f00

    __builtin_nanf32("")               0x7fc00000
    __builtin_nansf32("")              0x7fa00000
    quiet32(__builtin_nanf32(""))      0x7fc00000
    quiet32(__builtin_nansf32(""))     0x7fe00000

    __builtin_nanf64("")               0x7ff8000000000000
    __builtin_nansf64("")              0x7ff4000000000000
    quiet64(__builtin_nanf64(""))      0x7ff8000000000000
    quiet64(__builtin_nansf64(""))     0x7ffc000000000000

    __builtin_nanf128("")              0x7fff8000000000000000000000000000
    __builtin_nansf128("")             0x7fff4000000000000000000000000000
    quiet128(__builtin_nanf128(""))    0x7fff8000000000000000000000000000
    quiet128(__builtin_nansf128(""))   0x7fffc000000000000000000000000000

MIPS targets, however, print the following when built with GCC:

    bf16 not supported

    f16 not supported

    __builtin_nanf32("")               0x7fbfffff
    __builtin_nansf32("")              0x7fffffff
    quiet32(__builtin_nanf32(""))      0x7fbfffff
    quiet32(__builtin_nansf32(""))     0x7fbfffff

    __builtin_nanf64("")               0x7ff7ffffffffffff
    __builtin_nansf64("")              0x7fffffffffffffff
    quiet64(__builtin_nanf64(""))      0x7ff7ffffffffffff
    quiet64(__builtin_nansf64(""))     0x7ff7ffffffffffff

    __builtin_nanf128("")              0x7fff7fffffffffffffffffffffffffff
    __builtin_nansf128("")             0x7fffffffffffffffffffffffffffffff
    quiet128(__builtin_nanf128(""))    0x7fff7fffffffffffffffffffffffffff
    quiet128(__builtin_nansf128(""))   0x7fff7fffffffffffffffffffffffffff

(Note that GCC 13 changed the value of sNaNs on MIPS. Older versions reported
the same number as for qNaN.)

Clang just uses standard NaNs on MIPS for its `__builtin_` macros, but the
results after the quiet functions wind up the same as GCC due to hardware
instructions.

*/

#define __STDC_WANT_IEC_60559_EXT__
#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include <inttypes.h>

/* Fill C23 float names if not supported */

#if !defined(__FLT32_MANT_DIG__)
typedef float _Float32;
#endif

/* Clang defines _Float64 on mips but doesn't set __FLT64 macros for
 * some reason */
#if !defined(__FLT64_MANT_DIG__) && !(defined(__clang__) && defined(__mips__))
typedef double _Float64;
#endif

#if !defined(__FLT128_MANT_DIG__) && defined(__FLOAT128__)
#define __FLT128_MANT_DIG__
typedef __float128 _Float128;
#endif

#ifndef __builtin_nanf32
#define __builtin_nanf32 __builtin_nanf
#define __builtin_nansf32 __builtin_nansf
#endif
#ifndef __builtin_nanf64
#define __builtin_nanf64 __builtin_nan
#define __builtin_nansf64 __builtin_nans
#endif

/* Print reprs */

#ifdef __BFLT16_MANT_DIG__
/* Note that Clang never defines this bf macro */
typedef union { __bf16 f; uint16_t i; } cvtb16;
#endif
#ifdef __FLT16_MANT_DIG__
typedef union { _Float16 f; uint16_t i; } cvt16;
#endif
typedef union { _Float32 f; uint32_t i; } cvt32;
typedef union { _Float64 f; uint64_t i; } cvt64;
#ifdef __FLT128_MANT_DIG__
typedef union { _Float128 f; __int128 i; } cvt128;
#endif

#define pfb16(x) do { \
        cvtb16 c = { .f = x }; \
        printf("%-34s %#06"PRIx16"\n", #x, c.i); \
    } while (0)
#define pf16(x) do { \
        cvt16 c = { .f = x }; \
        printf("%-34s %#06"PRIx16"\n", #x, c.i); \
    } while (0)
#define pf32(x) do { \
        cvt32 c = { .f = x }; \
        printf("%-34s %#010"PRIx32"\n", #x, c.i); \
    } while (0)
#define pf64(x) do { \
        cvt64 c = { .f = x }; \
        printf("%-34s %#018"PRIx64"\n", #x, c.i); \
    } while (0)
#define pf128(x) do { \
        cvt128 c = { .f = x }; \
        uint64_t lo = c.i; \
        uint64_t hi = c.i >> 64; \
        printf("%-34s %#018"PRIx64"%016"PRIx64"\n", #x, hi, lo); \
    } while (0)

/* There are no macros for bf16 NaNs, so assume they're the
 * same as truncated f32 NaNs. */

#ifdef __BFLT16_MANT_DIG__
__bf16 bf_nan() {
    cvt32 c32 = { .f = __builtin_nanf32("") };
    cvtb16 c16 = { .i = c32.i >> 16 };
    return c16.f;
}

__bf16 bf_snan() {
    cvt32 c32 = { .f = __builtin_nansf32("") };
    cvtb16 c16 = { .i = c32.i >> 16 };
    return c16.f;
}
#endif

/* Use asm to hide operands and force runtime execution, since GCC replaces
 * sNaN * 1.0 with a canonical qNaN. */

#ifdef __BFLT16_MANT_DIG__
__bf16 quietb16(__bf16 x) {
    __bf16 y = 1.0;
    asm volatile("" : "+m"(x), "+m" (y) : :);
    return x * y;
}
#endif

#ifdef __FLT16_MANT_DIG__
_Float16 quiet16(_Float16 x) {
    _Float16 y = 1.0;
    asm volatile("" : "+m"(x), "+m" (y) : :);
    return x * y;
}
#endif

_Float32 quiet32(_Float32 x) {
    _Float32 y = 1.0;
    asm volatile("" : "+m"(x), "+m" (y) : :);
    return x * y;
}

_Float64 quiet64(_Float64 x) {
    _Float64 y = 1.0;
    asm volatile("" : "+m"(x), "+m" (y) : :);
    return x * y;
}

#ifdef __FLT128_MANT_DIG__
_Float128 quiet128(_Float128 x) {
    _Float128 y = 1.0;
    asm volatile("" : "+m"(x), "+m" (y) : :);
    return x * y;
}
#endif

int main() {
#ifdef __BFLT16_MANT_DIG__
    pfb16(bf_nan());
    pfb16(bf_snan());
    pfb16(quietb16(bf_nan()));
    pfb16(quietb16(bf_snan()));
#else
    printf("bf16 not supported\n");
#endif
    printf("\n");

#ifdef __FLT16_MANT_DIG__
    pf16(__builtin_nanf16(""));
    pf16(__builtin_nansf16(""));
    pf16(quiet16(__builtin_nanf16("")));
    pf16(quiet16(__builtin_nansf16("")));
#else
    printf("f16 not supported\n");
#endif
    printf("\n");

    pf32(__builtin_nanf32(""));
    pf32(__builtin_nansf32(""));
    pf32(quiet32(__builtin_nanf32("")));
    pf32(quiet32(__builtin_nansf32("")));
    printf("\n");

    pf64(__builtin_nanf64(""));
    pf64(__builtin_nansf64(""));
    pf64(quiet64(__builtin_nanf64("")));
    pf64(quiet64(__builtin_nansf64("")));
    printf("\n");

#ifdef __FLT128_MANT_DIG__
    pf128(__builtin_nanf128(""));
    pf128(__builtin_nansf128(""));
    pf128(quiet128(__builtin_nanf128("")));
    pf128(quiet128(__builtin_nansf128("")));
#else
    printf("f128 not supported\n");
#endif
}
