//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// issue #107899
// We end up relating `Const(ty: size_of<?0>, kind: Value(Branch([])))` with
// `Const(ty: size_of<T>, kind: Value(Branch([])))` which if you were to `==`
// the `ty` fields would return `false` and ICE. This test checks that we use
// actual semantic equality that takes into account aliases and infer vars.

use std::mem::size_of;

trait X<T> {
    fn f(self);
    fn g(self);
}

struct Y;

impl<T> X<T> for Y
where
    [(); size_of::<T>()]: Sized,
{
    fn f(self) {
        self.g();
    }
    fn g(self) {}
}

fn main() {}
