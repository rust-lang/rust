// issue: rust-lang/rust#114463
// ICE cannot convert `ReFree ..` to a region vid
#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete and may not be safe to use and/or cause compiler crashes
fn bug<'a>() {
    [(); (|_: &'a u8| (), 0).1];
    //~^ ERROR cannot capture late-bound lifetime in constant
}

pub fn main() {}
