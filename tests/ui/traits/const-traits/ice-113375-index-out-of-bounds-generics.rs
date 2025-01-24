//@ check-pass
//@ compile-flags: -Znext-solver
// effects ice https://github.com/rust-lang/rust/issues/113375 index out of bounds

#![allow(incomplete_features, unused)]
#![feature(adt_const_params)]

struct Bar<T>(T);

impl<T> Bar<T> {
    const fn value() -> usize {
        42
    }
}

struct Foo<const N: [u8; Bar::<u32>::value()]>;

pub fn main() {}
