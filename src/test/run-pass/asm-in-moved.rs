// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(asm)]

use std::cell::Cell;

#[repr(C)]
struct NoisyDrop<'a>(&'a Cell<&'static str>);
impl<'a> Drop for NoisyDrop<'a> {
    fn drop(&mut self) {
        self.0.set("destroyed");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    let status = Cell::new("alive");
    {
        let _y: Box<NoisyDrop>;
        let x = Box::new(NoisyDrop(&status));
        unsafe {
            asm!("mov $1, $0" : "=r"(_y) : "r"(x));
        }
        assert_eq!(status.get(), "alive");
    }
    assert_eq!(status.get(), "destroyed");
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
