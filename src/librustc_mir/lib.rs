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
#![deny(warnings)]
#![unstable(feature = "rustc_private", issue = "27812")]

#![feature(associated_consts)]
#![feature(box_patterns)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use] extern crate log;
extern crate graphviz as dot;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
#[macro_use]
#[no_link]
extern crate rustc_bitflags;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_const_math;
extern crate rustc_const_eval;

extern crate rustc_i128;

pub mod diagnostics;

pub mod build;
pub mod def_use;
pub mod graphviz;
mod hair;
pub mod mir_map;
pub mod pretty;
pub mod transform;

