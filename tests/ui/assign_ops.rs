// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(dead_code, unused_assignments)]
#[warn(clippy::assign_op_pattern)]
fn main() {
    let mut a = 5;
    a = a + 1;
    a = 1 + a;
    a = a - 1;
    a = a * 99;
    a = 42 * a;
    a = a / 2;
    a = a % 5;
    a = a & 1;
    a = 1 - a;
    a = 5 / a;
    a = 42 % a;
    a = 6 << a;
    let mut s = String::new();
    s = s + "bla";
}
