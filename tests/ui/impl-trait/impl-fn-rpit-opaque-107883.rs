//@ check-pass
// Regression test for <https;//github.com/rust-lang/rust/issues/107883>
#![feature(impl_trait_in_fn_trait_return)]
#![feature(unboxed_closures)] // only for `h`

use std::fmt::Debug;

fn f<T>() -> impl Fn(T) -> impl Debug {
    |_x| 15
}

fn g<T>() -> impl MyFn<(T,), Out = impl Debug> {
    |_x| 15
}

trait MyFn<T> {
    type Out;
}

impl<T, U, F: Fn(T) -> U> MyFn<(T,)> for F {
    type Out = U;
}

fn h<T>() -> impl Fn<(T,), Output = impl Debug> {
    |_x| 15
}

fn f_<T>() -> impl Fn(T) -> impl Debug {
    std::convert::identity(|_x| 15)
}

fn f__<T>() -> impl Fn(T) -> impl Debug {
    let r = |_x| 15;
    r
}

fn main() {}
