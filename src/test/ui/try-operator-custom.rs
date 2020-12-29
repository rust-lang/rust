// run-pass

#![feature(control_flow_enum)]
#![feature(never_type)]
#![feature(try_trait)]
#![feature(try_trait_v2)]

use std::convert::Infallible;
use std::ops::{BreakHolder, ControlFlow, Try2015, Try2021, Bubble};

enum MyResult<T, U> {
    Awesome(T),
    Terrible(U)
}

impl<U, V> Try2015 for MyResult<U, V> {
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

impl<U, V> Bubble for MyResult<U, V> {
    //type Continue = U;
    type Ok = U;
    type Holder = MyResult<Infallible, V>;
    fn continue_with(x: U) -> Self {
        MyResult::Awesome(x)
    }
    fn branch(self) -> ControlFlow<Self::Holder, U> {
        match self {
            MyResult::Awesome(u) => ControlFlow::Continue(u),
            MyResult::Terrible(e) => ControlFlow::Break(MyResult::Terrible(e)),
        }
    }
}

impl<U, V> BreakHolder<U> for MyResult<Infallible, V> {
    type Output = MyResult<U, V>;
}

impl<U, V, W: From<V>> Try2021<MyResult<Infallible, V>> for MyResult<U, W> {
    fn from_holder(x: MyResult<Infallible, V>) -> Self {
        match x {
            MyResult::Terrible(e) => MyResult::Terrible(From::from(e)),
            MyResult::Awesome(infallible) => match infallible {}
        }
    }
}

impl<U, V, W: From<V>> Try2021<Result<!, V>> for MyResult<U, W> {
    fn from_holder(x: Result<!, V>) -> Self {
        match x {
            Err(e) => MyResult::Terrible(From::from(e)),
            Ok(infallible) => match infallible {}
        }
    }
}

impl<U, V, W: From<V>> Try2021<MyResult<Infallible, V>> for Result<U, W> {
    fn from_holder(x: MyResult<Infallible, V>) -> Self {
        match x {
            MyResult::Terrible(e) => Err(From::from(e)),
            MyResult::Awesome(infallible) => match infallible {}
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
    let mapped = MyResult::<_, ()>::Awesome(4_i32).map(|x| x as i64);
    assert!(matches!(mapped, MyResult::Awesome(4_i64)));
}
