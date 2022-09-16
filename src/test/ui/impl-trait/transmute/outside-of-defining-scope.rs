// This previously compiled, but was intentionally changed in #101478.
// Lowered back to a future compat lint.
//
// See that PR for more details.
//
// check-pass
use std::mem::transmute;
fn foo() -> impl Sized {
    0u8
}

fn main() {
    unsafe {
        transmute::<_, u8>(foo());
        //~^ WARN relying on the underlying type of an opaque type in the type system
        //~| WARN this was previously accepted by the compiler but is being phased out
    }
}
