// run-pass
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]
// FIXME(const_generics_defaults): while we can allow this,
// we probably won't easily allow this with more complex const operations.
//
// So we have to make a conscious decision here when stabilizing a relaxed parameter ordering.
struct Foo<const N: usize, T = [u8; N]>(T);

impl<const N: usize> Foo<N> {
    fn new() -> Self {
        Foo([0; N])
    }
}

fn main() {
    assert_eq!(Foo::new().0, [0; 10]);
}
