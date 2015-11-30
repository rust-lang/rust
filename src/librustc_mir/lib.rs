// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Rust MIR: a lowered representation of Rust. Also: an experiment!

*/

#![crate_name = "rustc_mir"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]

#![feature(rustc_private)]
#![feature(into_cow)]

#[macro_use] extern crate log;
extern crate graphviz as dot;
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_front;
extern crate rustc_back;
extern crate syntax;

pub mod build;
pub mod mir_map;
mod hair;
mod graphviz;
pub mod transform;

