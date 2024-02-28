#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T> S<T> {
    type X = ()
    where
        T: Copy;
}

fn main() {
    let _: S::<String>::X; //~ ERROR trait `Copy` is not implemented for `String`
}
