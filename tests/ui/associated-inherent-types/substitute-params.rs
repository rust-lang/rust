//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T> S<T> {
    type P = T;
}

impl<T> S<(T,)> {
    type Un = T;
}

fn main() {
    // Regression test for issue #104240.
    let _: S<()>::P = ();

    // Regression test for issue #107468.
    let _: S<(i32,)>::Un = 0i32;
}
