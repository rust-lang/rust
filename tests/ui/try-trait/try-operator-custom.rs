//@ run-pass

#![feature(try_trait_v2)]

use std::ops::{ControlFlow, TryFromBreak, Try, TryAs};
use std::result::TryResult;

enum MyResult<T, U> {
    Awesome(T),
    Terrible(U),
}

impl<U, V> Try for MyResult<U, V> {
    type Kind = TryResult;
    type Output = U;
    type Break = V;

    fn from_output(u: U) -> MyResult<U, V> {
        MyResult::Awesome(u)
    }

    fn branch(self) -> ControlFlow<Self::Break, Self::Output> {
        match self {
            MyResult::Awesome(u) => ControlFlow::Continue(u),
            MyResult::Terrible(e) => ControlFlow::Break(e),
        }
    }
}

impl<T, U, V> TryAs<U> for MyResult<T, V> {
    type Try = MyResult<U, V>;
}

impl<T, E, F> TryFromBreak<E> for MyResult<T, F>
where
    E: Into<F>
{
    fn from_break(e: E) -> Self {
        MyResult::Terrible(e.into())
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
