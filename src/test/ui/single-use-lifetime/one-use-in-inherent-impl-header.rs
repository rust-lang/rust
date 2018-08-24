#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO warn for a lifetime used only once in an impl.
//
// (Actually, until #15872 is fixed, you can't use `'_` here, but
// hopefully that will come soon.)

struct Foo<'f> {
    data: &'f u32
}

impl<'f> Foo<'f> { //~ ERROR `'f` only used once
    fn inherent_a(&self) {
    }
}

fn main() { }
