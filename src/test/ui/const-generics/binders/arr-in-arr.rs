// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

struct Bar<'a, 'b, 'c>(&'a (), &'b (), &'c ());

trait Baz<'a, 'b, T> {}

struct Foo<'a>(&'a ()) where for<'b> [u8; {
    let _: Box<dyn for<'c> Baz<'a, 'b, [u8; {
        let _: Bar<'a, 'b, 'c>;
        3
    }]>>;
    4
}]:,; // FIXME(#79356): Add generic bounds here

fn main() {}
