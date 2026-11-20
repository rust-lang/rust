/* Reference side of `tests/run/overaligned_byval_abi.rs`, compiled by the real GCC.
 *
 * `Aligned` is an over-aligned aggregate passed by value ("byval"): the ABI places it in a stack
 * slot aligned to its own alignment, not packed right after the preceding argument. cg_gcc used
 * to build the GCC struct type from the field list alone, which dropped Rust's `repr(align(64))`,
 * so it placed the argument at an offset nobody else agreed on.
 *
 * The two functions here check both directions: `c_take_both` is a GCC-built callee for a cg_gcc
 * caller, and `c_call_rust` is a GCC-built caller for a cg_gcc callee.
 *
 * The checks are on the *values* received rather than on the address of the argument: which
 * alignment the ABI gives a stack slot is target-specific, but caller and callee agreeing on it
 * is not. A disagreement makes the arguments arrive as garbage. */

#include <stdint.h>

struct Big {
    int64_t a, b, c;
};

struct __attribute__((aligned(64))) Aligned {
    int32_t x;
};

/* Defined on the Rust side. */
extern int32_t rust_take_both(struct Big first, struct Aligned second, struct Big third,
                              struct Aligned fourth);

/* Called from Rust: checks what a cg_gcc caller passed. */
int32_t c_take_both(struct Big first, struct Aligned second, struct Big third,
                    struct Aligned fourth)
{
    if (first.a != 1 || first.b != 2 || first.c != 3)
        return 1;
    if (second.x != 42)
        return 2;
    if (third.a != 4 || third.b != 5 || third.c != 6)
        return 3;
    if (fourth.x != 43)
        return 4;
    return 0;
}

/* Called from Rust: passes the arguments the way the ABI says, for a cg_gcc callee to read. */
int32_t c_call_rust(void)
{
    struct Big first = {1, 2, 3};
    struct Big third = {4, 5, 6};
    struct Aligned second, fourth;

    second.x = 42;
    fourth.x = 43;
    return rust_take_both(first, second, third, fourth);
}
