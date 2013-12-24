// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
#[crate_id="crateresolve4b#0.1"];
#[crate_type = "lib"];

extern mod crateresolve4a = "crateresolve4a#0.2";

pub fn f() -> int { crateresolve4a::g() }
