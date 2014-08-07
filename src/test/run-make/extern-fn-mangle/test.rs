// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[no_mangle]
pub extern "C" fn foo() -> i32 { 3 }

#[no_mangle]
pub extern "C" fn bar() -> i32 { 5 }

#[link(name = "test", kind = "static")]
extern {
    fn add() -> i32;
}

fn main() {
    let back = unsafe { add() };
    assert_eq!(8, back);
}
