// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod bar {
    #[abi = "cdecl"]
    #[nolink]
    extern {}
}

mod zed {
    #[abi = "cdecl"]
    #[nolink]
    extern {}
}

mod libc {
    use std::libc::{c_int, c_void, size_t, ssize_t};

    #[abi = "cdecl"]
    #[nolink]
    extern {
        pub fn write(fd: c_int, buf: *c_void, count: size_t) -> ssize_t;
    }
}

mod baz {
    #[abi = "cdecl"]
    #[nolink]
    extern {}
}

pub fn main() { }
