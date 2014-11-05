// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = "\x80"; //~ ERROR may only be used
    let y = "\xff"; //~ ERROR may only be used
    let z = "\xe2"; //~ ERROR may only be used
    let a = b"\x00e2";  // ok because byte literal
}

