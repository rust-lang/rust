// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

trait A<T> {
    fn hey(v: T) -> Self;
}

impl<'a> A<&'a [u8; 3 + 4]> for &'a u32 {
    fn hey(_: &[u8; 3 + 4]) -> &u32 {
        &7
    }
}

fn main() {}
