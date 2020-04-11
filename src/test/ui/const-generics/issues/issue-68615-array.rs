// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

struct Foo<const V: [usize; 0] > {}

type MyFoo = Foo<{ [] }>;

fn main() {
    let _ = Foo::<{ [] }> {};
}
