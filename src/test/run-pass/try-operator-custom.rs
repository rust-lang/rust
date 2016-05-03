// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(question_mark, question_mark_carrier)]

use std::ops::Carrier;

enum MyResult<T, U> {
    Awesome(T),
    Terrible(U)
}

impl<U, V> Carrier for MyResult<U, V> {
    type Success = U;
    type Error = V;

    fn from_success(u: U) -> MyResult<U, V> {
        MyResult::Awesome(u)
    }

    fn from_error(e: V) -> MyResult<U, V> {
        MyResult::Terrible(e)
    }

    fn translate<T>(self) -> T
        where T: Carrier<Success=U, Error=V>
    {
        match self {
            MyResult::Awesome(u) => T::from_success(u),
            MyResult::Terrible(e) => T::from_error(e),
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
