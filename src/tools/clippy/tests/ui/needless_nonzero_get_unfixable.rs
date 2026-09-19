//@no-rustfix: the suggestion would remove the comment before `.get()`

#![warn(clippy::needless_nonzero_get)]

use std::num::NonZero;

fn main() {
    let nz = NonZero::new(1u32).unwrap();
    let mut value = 10u32;

    // This comment must not be removed by an automatic fix.
    let _ = nz /* keep this comment */
        .get()
        //~^ needless_nonzero_get
        .leading_zeros();

    // The operator suggestions share the same removal span, so they must preserve comments too.
    let _ = value
        / nz /* keep this comment */
            .get();
    //~^ needless_nonzero_get

    value %= nz /* keep this comment */
        .get();
    //~^ needless_nonzero_get
}
