// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr;

// check that we do not report a type like this as uninstantiable,
// even though it would be if the nxt field had type @foo:
struct foo(X);

struct X { x: uint, nxt: *const foo }

pub fn main() {
    let _x = foo(X {x: 0, nxt: ptr::null()});
}
