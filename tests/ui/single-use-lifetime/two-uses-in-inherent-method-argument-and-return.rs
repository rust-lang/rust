// Test that we DO NOT warn for a lifetime used twice in an impl method and
// header.

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'f> {
    data: &'f u32
}

impl<'f> Foo<'f> { //~ ERROR `'f` only used once
    fn inherent_a<'a>(&self, data: &'a u32) -> &'a u32{
      data
    }
}

fn main() { }
