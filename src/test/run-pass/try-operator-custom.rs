#![feature(try_trait)]

use std::ops::Try;

enum MyResult<T, U> {
    Awesome(T),
    Terrible(U)
}

impl<U, V> Try for MyResult<U, V> {
    type Ok = U;
    type Error = V;

    fn from_ok(u: U) -> MyResult<U, V> {
        MyResult::Awesome(u)
    }

    fn from_error(e: V) -> MyResult<U, V> {
        MyResult::Terrible(e)
    }

    fn into_result(self) -> Result<U, V> {
        match self {
            MyResult::Awesome(u) => Ok(u),
            MyResult::Terrible(e) => Err(e),
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
