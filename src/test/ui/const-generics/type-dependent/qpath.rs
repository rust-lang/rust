// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

struct A;
impl A {
    fn foo<const N: usize>() -> usize { N + 1 }
}

fn main() {
    assert_eq!(A::foo::<7>(), 8);
}
