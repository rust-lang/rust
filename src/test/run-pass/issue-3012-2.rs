// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:issue-3012-1.rs

#[feature(managed_boxes)];

extern crate socketlib;

use socketlib::socket;
use std::libc;

pub fn main() {
    let fd: libc::c_int = 1 as libc::c_int;
    let _sock = @socket::socket_handle(fd);
}
