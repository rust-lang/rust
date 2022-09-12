// This previously caused a query cycle which changed in #101478.
//
// See that PR for more details.
use std::mem::transmute;
fn foo() -> impl Sized {
    unsafe {
        transmute::<_, u8>(foo());
        //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
    }
    0u8
}

fn main() {}
