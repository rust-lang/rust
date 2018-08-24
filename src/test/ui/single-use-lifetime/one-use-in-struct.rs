// Test that we do not warn for named lifetimes in structs,
// even when they are only used once (since to not use a named
// lifetime is illegal!)
//
// compile-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'f> {
    data: &'f u32
}

enum Bar<'f> {
    Data(&'f u32)
}

trait Baz<'f> { }

fn main() { }
