// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id="boot#0.1"];
#[crate_type="lib"];
#[no_uv];

extern mod native;

#[no_mangle] // this needs to get called from C
pub extern "C" fn foo(argc: int, argv: **u8) -> int {
    do native::start(argc, argv) {
        do spawn {
            println!("hello");
        }
    }
}
