#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO NOT warn for a lifetime used just once in a return type,
// where that return type is in an inherent method.

struct Foo<'f> {
    data: &'f u32
}

impl<'f> Foo<'f> { //~ ERROR `'f` only used once
    fn inherent_a<'a>(&self) -> &'a u32 { // OK for 'a
        &22
    }
}

fn main() { }
