// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unordered containers, implemented as hash-tables

mod bench;
mod table;
#[cfg(stage0)]
#[path = "map_stage0.rs"]
pub mod map;
#[cfg(not(stage0))]
pub mod map;
#[cfg(stage0)]
#[path = "set_stage0.rs"]
pub mod set;
#[cfg(not(stage0))]
pub mod set;
pub mod state;
