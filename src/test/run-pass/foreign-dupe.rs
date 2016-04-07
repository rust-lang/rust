// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// calling pin_thread and that's having weird side-effects.

#![feature(libc)]

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

mod rustrt3 {
    // Different type, but same ABI (on all supported platforms).
    // Ensures that we don't ICE or trigger LLVM asserts when
    // importing the same symbol under different types.
    // See https://github.com/rust-lang/rust/issues/32740.
    extern {
        pub fn rust_get_test_int() -> *const u8;
    }
}

pub fn main() {
    unsafe {
        let x = rustrt1::rust_get_test_int();
        assert_eq!(x, rustrt2::rust_get_test_int());
        assert_eq!(x as *const _, rustrt3::rust_get_test_int());
    }
}
