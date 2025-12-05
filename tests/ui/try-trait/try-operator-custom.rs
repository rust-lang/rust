//@ run-pass

#![feature(try_trait_v2)]

use std::ops::{ControlFlow, FromResidual, Try};

enum MyResult<T, U> {
    Awesome(T),
    Terrible(U)
}

enum Never {}

impl<U, V> Try for MyResult<U, V> {
    type Output = U;
    type Residual = MyResult<Never, V>;

    fn from_output(u: U) -> MyResult<U, V> {
        MyResult::Awesome(u)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            MyResult::Awesome(u) => ControlFlow::Continue(u),
            MyResult::Terrible(e) => ControlFlow::Break(MyResult::Terrible(e)),
        }
    }
}

impl<U, V, W> FromResidual<MyResult<Never, V>> for MyResult<U, W> where V: Into<W> {
    fn from_residual(x: MyResult<Never, V>) -> Self {
        match x {
            MyResult::Terrible(e) => MyResult::Terrible(e.into()),
        }
    }
}

type ResultResidual<E> = Result<std::convert::Infallible, E>;

impl<U, V, W> FromResidual<ResultResidual<V>> for MyResult<U, W> where V: Into<W> {
    fn from_residual(x: ResultResidual<V>) -> Self {
        match x {
            Err(e) => MyResult::Terrible(e.into()),
        }
    }
}

impl<U, V, W> FromResidual<MyResult<Never, V>> for Result<U, W> where V: Into<W> {
    fn from_residual(x: MyResult<Never, V>) -> Self {
        match x {
            MyResult::Terrible(e) => Err(e.into()),
        }
    }
}

fn f(x: i32) -> Result<i32, String> {
    if x == 0 {
        Ok(42)
    } else {
        let y = g(x)?;
        Ok(y)
    }
}

fn g(x: i32) -> MyResult<i32, String> {
    let _y = f(x - 1)?;
    MyResult::Terrible("Hello".to_owned())
}

fn h() -> MyResult<i32, String> {
    let a: Result<i32, &'static str> = Err("Hello");
    let b = a?;
    MyResult::Awesome(b)
}

fn i() -> MyResult<i32, String> {
    let a: MyResult<i32, &'static str> = MyResult::Terrible("Hello");
    let b = a?;
    MyResult::Awesome(b)
}

fn main() {
    assert!(f(0) == Ok(42));
    assert!(f(10) == Err("Hello".to_owned()));
    let _ = h();
    let _ = i();
}
