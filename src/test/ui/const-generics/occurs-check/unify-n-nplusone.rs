#![feature(const_generics)]
#![allow(incomplete_features)]

// This test would try to unify `N` with `N + 1` which must fail the occurs check.

fn bind<const N: usize>(value: [u8; N]) -> [u8; N + 1] {
    //~^ ERROR constant expression depends on a generic parameter
    todo!()
}

fn sink(_: [u8; 5]) {}

fn main() {
    let mut arr = Default::default();
    arr = bind(arr);
    sink(arr);
}
