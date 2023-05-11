#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T> S<T> {
    type X = ()
    where
        T: Copy;
}

fn main() {
    let _: S::<String>::X; //~ ERROR the trait bound `String: Copy` is not satisfied
}
