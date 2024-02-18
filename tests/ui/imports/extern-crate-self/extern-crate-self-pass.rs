//@ build-pass (FIXME(62277): could be check-pass?)

extern crate self as foo;

struct S;

mod m {
    fn check() {
        foo::S; // OK
    }
}

fn main() {}
