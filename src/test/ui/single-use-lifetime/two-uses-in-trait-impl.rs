// Test that we DO NOT warn for a lifetime on an impl used in both
// header and in an associated type.
//
// build-pass (FIXME(62277): could be check-pass?)

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'f> {
    data: &'f u32
}

impl<'f> Iterator for Foo<'f> {
    type Item = &'f u32;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() { }
