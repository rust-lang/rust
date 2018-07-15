// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This note is annotated because the purpose of the test
// is to ensure that certain other notes are not generated.
#![deny(unused_unsafe)] //~ NOTE

// (test that no note is generated on this unsafe fn)
pub unsafe fn a() {
    fn inner() {
        unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
                                     //~^ NOTE
    }

    inner()
}

pub fn b() {
    // (test that no note is generated on this unsafe block)
    unsafe {
        fn inner() {
            unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
                                         //~^ NOTE
        }

        let () = ::std::mem::uninitialized();

        inner()
    }
}

fn main() {}
