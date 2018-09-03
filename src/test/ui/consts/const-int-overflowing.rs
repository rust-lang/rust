// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x: &'static (i32, bool) = &(5_i32.overflowing_add(3)); //~ ERROR does not live long enough
    let y: &'static (i32, bool) = &(5_i32.overflowing_sub(3)); //~ ERROR does not live long enough
    let z: &'static (i32, bool) = &(5_i32.overflowing_mul(3)); //~ ERROR does not live long enough
}
