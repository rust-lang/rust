// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: matchck eval1 eval2

#[cfg(matchck)]
const X: i32 = { let 0 = 0; 0 };
//[matchck]~^ ERROR refutable pattern in local binding

#[cfg(matchck)]
static Y: i32 = { let 0 = 0; 0 };
//[matchck]~^ ERROR refutable pattern in local binding

#[cfg(matchck)]
trait Bar {
    const X: i32 = { let 0 = 0; 0 };
    //[matchck]~^ ERROR refutable pattern in local binding
}

#[cfg(matchck)]
impl Bar for () {
    const X: i32 = { let 0 = 0; 0 };
    //[matchck]~^ ERROR refutable pattern in local binding
}

#[cfg(eval1)]
enum Foo {
    A = { let 0 = 0; 0 },
    //[eval1]~^ ERROR refutable pattern in local binding
}

fn main() {
    #[cfg(eval2)]
    let x: [i32; { let 0 = 0; 0 }] = [];
    //[eval2]~^ ERROR refutable pattern in local binding
}
