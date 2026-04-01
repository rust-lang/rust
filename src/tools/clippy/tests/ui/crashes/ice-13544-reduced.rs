//@ check-pass
#![warn(clippy::significant_drop_tightening)]
#![allow(unused, clippy::no_effect)]

use std::marker::PhantomData;

trait Trait {
    type Assoc: Trait;
}
struct S<T: Trait>(*const S<T::Assoc>, PhantomData<T>);

fn f<T: Trait>(x: &mut S<T>) {
    &mut x.0;
}

fn main() {}
