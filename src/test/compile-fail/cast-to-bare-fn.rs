// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(_x: int) { }

#[fixed_stack_segment]
fn main() {
    let v: u64 = 5;
    let x = foo as extern "C" fn() -> int;
    //~^ ERROR non-scalar cast
    let y = v as extern "Rust" fn(int) -> (int, int);
    //~^ ERROR non-scalar cast
    y(x());
}
