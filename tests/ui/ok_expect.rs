// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

struct MyError(()); // doesn't implement Debug

#[derive(Debug)]
struct MyErrorWithParam<T> {
    x: T,
}

fn main() {
    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();

    res.ok().expect("disaster!");
    // the following should not warn, since `expect` isn't implemented unless
    // the error type implements `Debug`
    let res2: Result<i32, MyError> = Ok(0);
    res2.ok().expect("oh noes!");
    let res3: Result<u32, MyErrorWithParam<u8>> = Ok(0);
    res3.ok().expect("whoof");
    let res4: Result<u32, io::Error> = Ok(0);
    res4.ok().expect("argh");
    let res5: io::Result<u32> = Ok(0);
    res5.ok().expect("oops");
    let res6: Result<u32, &str> = Ok(0);
    res6.ok().expect("meh");
}
