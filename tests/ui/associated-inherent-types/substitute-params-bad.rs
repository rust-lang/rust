// Regression test for issue #105305 and for
// https://github.com/rust-lang/rust/issues/107468#issuecomment-1409096700

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T, 'a> S<T> { //~ ERROR lifetime parameters must be declared prior to type and const parameters
    type P = T;
}

struct Subj<T>(T);

impl<T, S> Subj<(T, S)> {
    type Un = (T, S);
}

fn main() {
    type A = S<()>::P;

    let _: Subj<(i32, i32)>::Un = 0i32; //~ ERROR mismatched types
}
