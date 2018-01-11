// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
fn main() {
    let x: *const _ = 0 as _; //~ ERROR cannot cast

    let x: *const _ = 0 as *const _; //~ ERROR cannot cast
    let y: Option<*const fmt::Debug> = Some(x) as _;

    let x = 0 as *const i32 as *const _ as *mut _; //~ ERROR cannot cast
}
