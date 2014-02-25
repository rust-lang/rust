// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id="boot#0.1"];
#[crate_type="dylib"];
#[no_uv];

extern crate rustuv;
extern crate green;

use std::rt::crate_map::{CrateMap, rust_set_crate_map};

// pull in this symbol from libstd into this crate (for convenience)
#[no_mangle]
pub static set_crate_map: extern "C" fn(*CrateMap<'static>) = rust_set_crate_map;

#[no_mangle] // this needs to get called from C
pub extern "C" fn foo(argc: int, argv: **u8) -> int {
    green::start(argc, argv, proc() {
        if log_enabled!(std::logging::DEBUG) { return }
        fail!()
    })
}
