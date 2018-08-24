#![feature(alloc)]
#![allow(unused_extern_crates)]

extern crate alloc;

mod foo {
    pub trait alloc {
        fn do_something();
    }
}

use foo::alloc;
//~^ ERROR E0254

fn main() {}
