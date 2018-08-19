// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

trait Mirror {
    type It;
}

impl<T> Mirror for T {
    type It = Self;
}


#[rustc_error]
fn main() { //~ ERROR compilation successful
    let c: <u32 as Mirror>::It = 5;
    const CCCC: <u32 as Mirror>::It = 5;
}
