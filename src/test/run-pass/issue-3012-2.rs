// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-3012-1.rs

// pretty-expanded FIXME #23616

extern crate socketlib;

use socketlib::socket;

pub fn main() {
    let fd: u32 = 1 as u32;
    let _sock: Box<_> = Box::new(socket::socket_handle(fd));
}
