/* Strong definitions of the functions that `tests/run/weak_function_linkage.rs` also defines, but
 * weakly. The linker has to keep these and drop the Rust ones.
 *
 * A backend that emits the Rust definitions as ordinary global symbols does not merely pick the
 * wrong one: the link fails outright with a duplicate definition. */

#include <stdint.h>

int32_t weak_function(void)
{
    return 1;
}

int32_t weak_odr_function(void)
{
    return 2;
}

int32_t linkonce_function(void)
{
    return 3;
}

int32_t linkonce_odr_function(void)
{
    return 4;
}

int32_t weak_inline_function(void)
{
    return 8;
}

/* `available_externally` promises the real definition lives elsewhere: a backend may call this one
 * or emit an equivalent copy of the Rust body, so the two have to return the same value. */
int32_t available_externally_function(void)
{
    return 7;
}

/* Called from Rust, so that the calls also go through a caller that GCC compiled: a cg_gcc caller
 * could inline the weak body it can see instead of calling the symbol. */
int32_t c_call_all(void)
{
    if (weak_function() != 1)
        return 11;
    if (weak_odr_function() != 2)
        return 12;
    if (linkonce_function() != 3)
        return 13;
    if (linkonce_odr_function() != 4)
        return 14;
    return 0;
}
