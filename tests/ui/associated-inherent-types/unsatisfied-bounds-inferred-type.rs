#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T: Copy> S<T> {
    type T = T;
}

fn main() {
    let _: S<_>::T = String::new(); //~ ERROR trait `Copy` is not implemented for `String`
}
