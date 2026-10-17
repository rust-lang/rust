// Used to cause a query cycle due to using `TypingEnv::PostAnalysis`,
// in #119821 const eval was changed to always use this mode.
//
//@ check-pass

use std::mem::transmute;

fn foo() -> impl Sized {
    //~^ WARN function cannot return without recursing
    unsafe {
        transmute::<_, u8>(foo());
    }
    0u8
}

fn main() {}
