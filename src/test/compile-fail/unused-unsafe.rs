// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Exercise the unused_unsafe attribute in some positive and negative cases

#[deny(unused_unsafe)];

use core::libc;

fn callback<T>(_f: &fn() -> T) -> T { fail!() }

fn bad1() { unsafe {} }                  //~ ERROR: unnecessary "unsafe" block
fn bad2() { unsafe { bad1() } }          //~ ERROR: unnecessary "unsafe" block
unsafe fn bad3() {}                      //~ ERROR: unnecessary "unsafe" function
unsafe fn bad4() { unsafe {} }           //~ ERROR: unnecessary "unsafe" function
                                         //~^ ERROR: unnecessary "unsafe" block
fn bad5() { unsafe { do callback {} } }  //~ ERROR: unnecessary "unsafe" block

unsafe fn good0() { libc::exit(1) }
fn good1() { unsafe { libc::exit(1) } }
fn good2() {
    /* bug uncovered when implementing warning about unused unsafe blocks. Be
       sure that when purity is inherited that the source of the unsafe-ness
       is tracked correctly */
    unsafe {
        unsafe fn what() -> ~[~str] { libc::exit(2) }

        do callback {
            what();
        }
    }
}

#[allow(unused_unsafe)] unsafe fn allowed0() {}
#[allow(unused_unsafe)] fn allowed1() { unsafe {} }

fn main() { }
