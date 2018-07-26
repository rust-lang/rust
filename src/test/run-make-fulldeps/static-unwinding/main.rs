// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate lib;

use std::thread;

static mut statik: isize = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe { statik = 1; }
    }
}

fn main() {
    thread::spawn(move|| {
        let _a = A;
        lib::callback(|| panic!());
    }).join().unwrap_err();

    unsafe {
        assert_eq!(lib::statik, 1);
        assert_eq!(statik, 1);
    }
}
