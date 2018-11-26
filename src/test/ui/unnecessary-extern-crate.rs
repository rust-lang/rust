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

extern crate libc;
//~^ ERROR unused extern crate
//~| HELP remove
extern crate libc as x;
//~^ ERROR unused extern crate
//~| HELP remove

extern crate proc_macro;

#[macro_use]
extern crate test;

pub extern crate test as y;

pub extern crate alloc;

pub(crate) extern crate alloc as a;

crate extern crate alloc as b;

mod foo {
    pub(in crate::foo) extern crate alloc as c;

    pub(super) extern crate alloc as d;

    extern crate libc;
    //~^ ERROR unused extern crate
    //~| HELP remove

    extern crate libc as x;
    //~^ ERROR unused extern crate
    //~| HELP remove

    pub extern crate test;

    pub extern crate test as y;

    mod bar {
        extern crate libc;
        //~^ ERROR unused extern crate
        //~| HELP remove

        extern crate libc as x;
        //~^ ERROR unused extern crate
        //~| HELP remove

        pub(in crate::foo::bar) extern crate alloc as e;

        fn dummy() {
            e::string::String::new();
        }
    }

    fn dummy() {
        c::string::String::new();
        d::string::String::new();
    }
}


fn main() {
    a::string::String::new();
    b::string::String::new();

    proc_macro::TokenStream::new();
}
