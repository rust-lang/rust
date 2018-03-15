// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = { (&FOO as *const i32) == (&BAR as *const i32) }; //~ ERROR E0395
                   //~| NOTE comparing raw pointers in static
fn main() {
}
