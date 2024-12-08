// This causes a query cycle due to using `TypingEnv::PostAnalysis`,
// in #119821 const eval was changed to always use this mode.
//
// See that PR for more details.
use std::mem::transmute;
fn foo() -> impl Sized {
    //~^ ERROR cycle detected when computing type of
    //~| WARN function cannot return without recursing
    unsafe {
        transmute::<_, u8>(foo());
    }
    0u8
}

fn main() {}
