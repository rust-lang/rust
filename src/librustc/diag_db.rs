// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The diagnostic database.
//!
//! Extended information about Rust diagnostics is included in the
//! diag_db.md file and can be loaded at runtime with the `load`
//! function.

#[cfg(not(stage0))]

pub fn load() -> ~[(&'static str, &'static str, &'static str)] {
    ~[include!("diag_db.md")]
}
