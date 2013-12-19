// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id="crateresolve_calories#0.1"];
// NOTE: remove after the next snapshot
#[link(name = "crateresolve_calories",
       vers = "0.1",
       calories = "200")];

#[crate_type = "lib"];

pub fn f() -> int { 200 }
