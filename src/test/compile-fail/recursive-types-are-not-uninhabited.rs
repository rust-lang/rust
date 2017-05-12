// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//#![feature(never_type)]

struct R<'a> {
    r: &'a R<'a>,
}

fn foo(res: Result<u32, &R>) -> u32 {
    let Ok(x) = res;
    //~^ ERROR refutable pattern
    x
}

fn main() {
    foo(Ok(23));
}

