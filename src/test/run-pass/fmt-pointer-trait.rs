// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(libc)]
extern crate libc;
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let p: *const libc::c_void = ptr::null();
    let rc = Rc::new(1usize);
    let arc = Arc::new(1usize);
    let b = Box::new("hi");

    let _ = format!("{:p}{:p}{:p}",
                    rc, arc, b);

    if cfg!(target_pointer_width = "32") {
        assert_eq!(format!("{:#p}", p),
                   "0x00000000");
    } else {
        assert_eq!(format!("{:#p}", p),
                   "0x0000000000000000");
    }
    assert_eq!(format!("{:p}", p),
               "0x0");
}
