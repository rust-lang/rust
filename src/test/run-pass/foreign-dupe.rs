// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// calling pin_task and that's having wierd side-effects.

mod rustrt1 {
    extern crate libc;

    #[link(name = "rust_test_helpers")]
    extern {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

mod rustrt2 {
    extern crate libc;

    extern {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() {
    unsafe {
        rustrt1::rust_get_test_int();
        rustrt2::rust_get_test_int();
    }
}
