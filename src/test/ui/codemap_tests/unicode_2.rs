// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(non_ascii_idents)]

fn main() {
    let _ = ("a̐éö̲", 0u7); //~ ERROR invalid width
    let _ = ("아あ", 1i42); //~ ERROR invalid width
    let _ = a̐é; //~ ERROR cannot find
}
