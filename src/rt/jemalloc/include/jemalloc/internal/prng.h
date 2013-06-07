/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

/*
 * Simple linear congruential pseudo-random number generator:
 *
 *   prng(y) = (a*x + c) % m
 *
 * where the following constants ensure maximal period:
 *
 *   a == Odd number (relatively prime to 2^n), and (a-1) is a multiple of 4.
 *   c == Odd number (relatively prime to 2^n).
 *   m == 2^32
 *
 * See Knuth's TAOCP 3rd Ed., Vol. 2, pg. 17 for details on these constraints.
 *
 * This choice of m has the disadvantage that the quality of the bits is
 * proportional to bit position.  For example. the lowest bit has a cycle of 2,
 * the next has a cycle of 4, etc.  For this reason, we prefer to use the upper
 * bits.
 *
 * Macro parameters:
 *   uint32_t r          : Result.
 *   unsigned lg_range   : (0..32], number of least significant bits to return.
 *   uint32_t state      : Seed value.
 *   const uint32_t a, c : See above discussion.
 */
#define prng32(r, lg_range, state, a, c) do {				\
    assert(lg_range > 0);						\
    assert(lg_range <= 32);						\
                                    \
    r = (state * (a)) + (c);					\
    state = r;							\
    r >>= (32 - lg_range);						\
} while (false)

/* Same as prng32(), but 64 bits of pseudo-randomness, using uint64_t. */
#define prng64(r, lg_range, state, a, c) do {				\
    assert(lg_range > 0);						\
    assert(lg_range <= 64);						\
                                    \
    r = (state * (a)) + (c);					\
    state = r;							\
    r >>= (64 - lg_range);						\
} while (false)

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
