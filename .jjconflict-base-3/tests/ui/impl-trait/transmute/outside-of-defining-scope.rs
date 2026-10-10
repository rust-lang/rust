//! Check that typeck can observe the size of an opaque type.
//@ check-pass
use std::mem::transmute;
fn foo() -> impl Sized {
    0u8
}

fn main() {
    unsafe {
        transmute::<_, u8>(foo());
    }
}
