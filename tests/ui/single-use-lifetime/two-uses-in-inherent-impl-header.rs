// Test that we DO NOT warn for a lifetime used twice in an impl.
//
// check-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'f> {
    data: &'f u32,
}

impl<'f> Foo<'f> {
    fn inherent_a(&self, data: &'f u32) {}
}

fn main() {}
