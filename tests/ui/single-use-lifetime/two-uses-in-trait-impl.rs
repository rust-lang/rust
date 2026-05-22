// Test that we DO NOT warn for a lifetime on an impl used in both
// header and in an associated type.
//
//@ check-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'f> {
    data: &'f u32,
}

impl<'f> Iterator for Foo<'f> {
    type Item = &'f u32;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

trait BoundTrait<'a> {
    fn foo(self, handler: &Handler<'a>);
}

struct Handler<'a>(fn(&'a u32));
struct Bar<'b>(&'b u32);

// https://github.com/rust-lang/rust/issues/153836
impl<'a, 'b: 'a> BoundTrait<'a> for Bar<'b> {
    fn foo(self, handler: &Handler<'a>) {
        (handler.0)(self.0);
    }
}

fn main() {}
