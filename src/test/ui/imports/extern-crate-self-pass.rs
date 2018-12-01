// compile-pass

#![feature(extern_crate_self)]

extern crate self as foo;

struct S;

mod m {
    fn check() {
        foo::S; // OK
    }
}

fn main() {}
