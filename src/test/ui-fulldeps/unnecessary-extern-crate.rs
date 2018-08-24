// edition:2018

#![deny(unused_extern_crates)]
#![feature(alloc, test, libc)]

extern crate alloc;
//~^ ERROR unused extern crate
//~| HELP remove
extern crate alloc as x;
//~^ ERROR unused extern crate
//~| HELP remove

#[macro_use]
extern crate test;

pub extern crate test as y;
//~^ ERROR `extern crate` is not idiomatic in the new edition
//~| HELP convert it to a `pub use`

pub extern crate libc;
//~^ ERROR `extern crate` is not idiomatic in the new edition
//~| HELP convert it to a `pub use`

pub(crate) extern crate libc as a;
//~^ ERROR `extern crate` is not idiomatic in the new edition
//~| HELP convert it to a `pub(crate) use`

crate extern crate libc as b;
//~^ ERROR `extern crate` is not idiomatic in the new edition
//~| HELP convert it to a `crate use`

mod foo {
    pub(in crate::foo) extern crate libc as c;
    //~^ ERROR `extern crate` is not idiomatic in the new edition
    //~| HELP convert it to a `pub(in crate::foo) use`

    pub(super) extern crate libc as d;
    //~^ ERROR `extern crate` is not idiomatic in the new edition
    //~| HELP convert it to a `pub(super) use`

    extern crate alloc;
    //~^ ERROR unused extern crate
    //~| HELP remove

    extern crate alloc as x;
    //~^ ERROR unused extern crate
    //~| HELP remove

    pub extern crate test;
    //~^ ERROR `extern crate` is not idiomatic in the new edition
    //~| HELP convert it

    pub extern crate test as y;
    //~^ ERROR `extern crate` is not idiomatic in the new edition
    //~| HELP convert it

    mod bar {
        extern crate alloc;
        //~^ ERROR unused extern crate
        //~| HELP remove

        extern crate alloc as x;
        //~^ ERROR unused extern crate
        //~| HELP remove

        pub(in crate::foo::bar) extern crate libc as e;
        //~^ ERROR `extern crate` is not idiomatic in the new edition
        //~| HELP convert it to a `pub(in crate::foo::bar) use`

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
}
