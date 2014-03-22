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

#![allow(dead_code)]
#![deny(unused_unsafe)]
#![allow(deprecated_owned_vector)]


mod foo {
    extern {
        pub fn bar();
    }
}

fn callback<T>(_f: || -> T) -> T { fail!() }
unsafe fn unsf() {}

fn bad1() { unsafe {} }                  //~ ERROR: unnecessary `unsafe` block
fn bad2() { unsafe { bad1() } }          //~ ERROR: unnecessary `unsafe` block
unsafe fn bad3() { unsafe {} }           //~ ERROR: unnecessary `unsafe` block
fn bad4() { unsafe { callback(||{}) } }  //~ ERROR: unnecessary `unsafe` block
unsafe fn bad5() { unsafe { unsf() } }   //~ ERROR: unnecessary `unsafe` block
fn bad6() {
    unsafe {                             // don't put the warning here
        unsafe {                         //~ ERROR: unnecessary `unsafe` block
            unsf()
        }
    }
}
unsafe fn bad7() {
    unsafe {                             //~ ERROR: unnecessary `unsafe` block
        unsafe {                         //~ ERROR: unnecessary `unsafe` block
            unsf()
        }
    }
}

unsafe fn good0() { unsf() }
fn good1() { unsafe { unsf() } }
fn good2() {
    /* bug uncovered when implementing warning about unused unsafe blocks. Be
       sure that when purity is inherited that the source of the unsafe-ness
       is tracked correctly */
    unsafe {
        unsafe fn what() -> Vec<~str> { fail!() }

        callback(|| {
            what();
        });
    }
}

unsafe fn good3() { foo::bar() }
fn good4() { unsafe { foo::bar() } }

#[allow(unused_unsafe)] fn allowed() { unsafe {} }

fn main() {}
