/* Reference side of `tests/run/sret.rs`, compiled by the real GCC.
 *
 * `Big` is returned in memory ("sret"): the caller passes a hidden pointer to the slot the
 * callee fills, and the callee returns that pointer. The two functions here check both
 * directions: `c_make_big` is a GCC-built callee for a cg_gcc caller, and `c_call_rust` is a
 * GCC-built caller for a cg_gcc callee.
 *
 * Which register carries the hidden pointer is target-specific, but the two sides of a call
 * agreeing on it is not: a backend that passes it as an ordinary first parameter instead
 * disagrees with GCC on every target that reserves a separate register for it. */

#include <stdint.h>

struct Big {
    int64_t a, b, c;
};

/* Defined on the Rust side. */
extern struct Big rust_make_big(int64_t a, int64_t b, int64_t c);

/* Called from Rust: a GCC-built callee for a cg_gcc caller. */
struct Big c_make_big(int64_t a, int64_t b, int64_t c)
{
    struct Big result = {a, b, c};
    return result;
}

/* Called from Rust: a GCC-built caller for a cg_gcc callee. */
int32_t c_call_rust(void)
{
    struct Big value = rust_make_big(50, 51, 52);

    if (value.a != 50 || value.b != 51 || value.c != 52)
        return 1;
    return 0;
}
