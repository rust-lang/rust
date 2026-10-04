// issue: rust-lang/rust#114463
// ICE cannot convert `ReFree ..` to a region vid
#![feature(generic_const_exprs)]

fn bug<'a>() {
    [(); (|_: &'a u8| (), 0).1];
    //~^ ERROR cannot capture late-bound lifetime in constant
}

pub fn main() {}
