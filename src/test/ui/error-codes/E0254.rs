#![allow(unused_extern_crates, non_camel_case_types)]

extern crate alloc;

mod foo {
    pub trait alloc {
        fn do_something();
    }
}

use foo::alloc;
//~^ ERROR E0254

fn main() {}
