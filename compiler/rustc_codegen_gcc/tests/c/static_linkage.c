/* Strong definitions of the statics that `tests/run/static_linkage.rs` also defines, but weakly.
 * The linker has to keep these and drop the Rust ones; a backend that emits the Rust definitions
 * as ordinary global symbols fails the link with a duplicate definition instead.
 *
 * `internal_static` is the opposite case: the Rust side keeps its own, and the two definitions
 * coexist because the Rust one is local. */

#include <stdint.h>

int32_t weak_static = 1;
int32_t weak_odr_static = 2;
int32_t linkonce_static = 3;
int32_t linkonce_odr_static = 4;
int32_t common_static = 5;
int32_t internal_static = 200;

/* `available_externally` promises the real definition lives elsewhere: a backend may read this one
 * or emit an equivalent copy of the Rust initializer, so the two have to hold the same value. */
int32_t available_externally_static = 7;

/* Called from Rust, so that the reads also happen in a translation unit GCC compiled. */
int32_t c_read_all(void)
{
    if (weak_static != 1)
        return 11;
    if (weak_odr_static != 2)
        return 12;
    if (linkonce_static != 3)
        return 13;
    if (linkonce_odr_static != 4)
        return 14;
    if (common_static != 5)
        return 15;
    if (internal_static != 200)
        return 16;
    return 0;
}
