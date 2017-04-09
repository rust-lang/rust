// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a regression found when building compiler. The `produce()`
// error type `T` winds up getting unified with result of `x.parse()`;
// the type of the closure given to `unwrap_or_else` needs to be
// inferred to `usize`.

use std::num::ParseIntError;

fn produce<T>() -> Result<&'static str, T> {
    Ok("22")
}

fn main() {
    let x: usize = produce()
        .and_then(|x| x.parse())
        .unwrap_or_else(|_| panic!());
    println!("{}", x);
}
