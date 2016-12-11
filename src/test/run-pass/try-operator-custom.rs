// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(question_mark, question_mark_qm, question_mark_try)]

use std::ops::QuestionMark;
use std::ops::Try;

#[derive(PartialEq)]
enum MyResult<U, V> {
    Awesome(U),
    Terrible(V)
}

impl<U, V, X, Y> QuestionMark<MyResult<X, Y>> for MyResult<U, V>
    where Y: From<V>,
{
    type Continue = U;
    fn question_mark(self) -> Try<Self::Continue, MyResult<X, Y>> {
        match self {
            MyResult::Awesome(u) => Try::Continue(u),
            MyResult::Terrible(e) => Try::Done(MyResult::Terrible(Y::from(e))),
        }
    }
}

fn f(x: i32) -> Result<i32, String> {
    if x == 0 {
        Ok(42)
    } else {
        let y = Err(String::new())?;
        Ok(y)
    }
}

fn g(x: i32) -> MyResult<i32, String> {
    if x == 0 {
        return MyResult::Awesome(42);
    } else {
        let _y = i()?;
        MyResult::Awesome(1)
    }
}

fn i() -> MyResult<u32, String> {
    let a: MyResult<u32, &'static str> = MyResult::Terrible("Hello");
    let b = a?;
    MyResult::Awesome(b)
}

fn main() {
    assert!(g(0) == MyResult::Awesome(42));
    assert!(g(10) == MyResult::Terrible("Hello".to_owned()));
    let _ = i();
}
