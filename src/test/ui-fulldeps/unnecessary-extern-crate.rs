// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// edition:2018

#![deny(unused_extern_crates)]
#![feature(alloc, test, libc, crate_visibility_modifier)]

extern crate alloc;
//~^ ERROR unused extern crate
//~| HELP remove
extern crate alloc as x;
//~^ ERROR unused extern crate
//~| HELP remove

extern crate proc_macro;

#[macro_use]
extern crate test;

pub extern crate test as y;

pub extern crate libc;

pub(crate) extern crate libc as a;

crate extern crate libc as b;

mod foo {
    pub(in crate::foo) extern crate libc as c;

    pub(super) extern crate libc as d;

    extern crate alloc;
    //~^ ERROR unused extern crate
    //~| HELP remove

    extern crate alloc as x;
    //~^ ERROR unused extern crate
    //~| HELP remove

    pub extern crate test;

    pub extern crate test as y;

    mod bar {
        extern crate alloc;
        //~^ ERROR unused extern crate
        //~| HELP remove

        extern crate alloc as x;
        //~^ ERROR unused extern crate
        //~| HELP remove

        pub(in crate::foo::bar) extern crate libc as e;

        fn dummy() {
            unsafe {
                e::getpid();
            }
        }
    }

    fn dummy() {
        unsafe {
            c::getpid();
            d::getpid();
        }
    }
}


fn main() {
    unsafe { a::getpid(); }
    unsafe { b::getpid(); }

    proc_macro::TokenStream::new();
}
