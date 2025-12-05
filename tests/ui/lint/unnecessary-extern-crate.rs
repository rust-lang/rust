//@ edition:2018

#![deny(unused_extern_crates)]
#![feature(test)]

extern crate core;
//~^ ERROR unused extern crate
//~| HELP remove
extern crate core as x;
//~^ ERROR unused extern crate
//~| HELP remove

extern crate proc_macro;

#[macro_use]
extern crate test;

pub extern crate test as y;

pub extern crate alloc;

pub(crate) extern crate alloc as a;

pub(crate) extern crate alloc as b;

mod foo {
    pub(in crate::foo) extern crate alloc as c;

    pub(super) extern crate alloc as d;

    extern crate core;
    //~^ ERROR unused extern crate
    //~| HELP remove

    extern crate core as x;
    //~^ ERROR unused extern crate
    //~| HELP remove

    pub extern crate test;

    pub extern crate test as y;

    mod bar {
        extern crate core;
        //~^ ERROR unused extern crate
        //~| HELP remove

        extern crate core as x;
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
