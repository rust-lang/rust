// This causes a query cycle due to incorrectly using `Reveal::All`,
// in #101478 this was changed to a future compat error which only
// triggers after the query cycle.
//
// See that PR for more details.
use std::mem::transmute;
fn foo() -> impl Sized {
    //~^ ERROR cycle detected when computing type of
    unsafe {
        transmute::<_, u8>(foo());
        //~^ ERROR cannot transmute between types of different sizes
    }
    0u8
}

fn main() {}
