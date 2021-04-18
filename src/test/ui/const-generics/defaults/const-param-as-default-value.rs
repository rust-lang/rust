// run-pass
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]
struct Foo<const N: usize, const M: usize = N>([u8; N], [u8; M]);

fn foo<const N: usize>() -> Foo<N> {
    let x = [0; N];
    Foo(x, x)
}

fn main() {
    let val = foo::<13>();
    assert_eq!(val.0, val.1);
}
