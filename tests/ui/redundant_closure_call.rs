// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::redundant_closure_call)]

fn main() {
    let a = (|| 42)();

    let mut i = 1;
    let mut k = (|m| m + 1)(i);

    k = (|a, b| a * b)(1, 5);

    let closure = || 32;
    i = closure();

    let closure = |i| i + 1;
    i = closure(3);

    i = closure(4);

    #[allow(clippy::needless_return)]
    (|| return 2)();
    (|| -> Option<i32> { None? })();
    (|| -> Result<i32, i32> { r#try!(Err(2)) })();
}
