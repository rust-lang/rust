//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T: O> S<T> {
    type P = <T as O>::P;
}

trait O {
    type P;
}

impl O for i32 {
    type P = String;
}

fn main() {
    let _: S<i32>::P = String::new();
}
