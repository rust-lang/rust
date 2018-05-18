// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: cfail1 cfail2 cfail3
// compile-flags: -Coverflow-checks=on
// compile-pass

#![allow(warnings)]
#![warn(const_err)]

fn main() {
    255u8 + 1; //~ WARNING this expression will panic at run-time
}
