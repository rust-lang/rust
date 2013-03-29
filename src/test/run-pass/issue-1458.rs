// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn plus_one(f: &fn() -> int) -> int {
  return f() + 1;
}

fn ret_plus_one() -> extern fn(&fn() -> int) -> int {
  return plus_one;
}

pub fn main() {
    let z = do (ret_plus_one()) || { 2 };
    assert!(z == 3);
}
