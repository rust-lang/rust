//@ run-pass

#![feature(try_trait_v2)]

use std::ops::Try;

fn monad_unit<T: Try>(x: <T as Try>::Output) -> T {
    T::from_output(x)
}

fn monad_bind<T1: Try<Residual = R>, T2: Try<Residual = R>, R>(
    mx: T1,
    f: impl FnOnce(<T1 as Try>::Output) -> T2)
-> T2 {
    let x = mx?;
    f(x)
}

fn main() {
    let mx: Option<i32> = monad_unit(1);
    let my = monad_bind(mx, |x| Some(x + 1));
    let mz = monad_bind(my, |x| Some(-x));
    assert_eq!(mz, Some(-2));
}
