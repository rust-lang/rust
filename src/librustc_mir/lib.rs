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

#![deny(warnings)]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(catch_expr)]
#![feature(conservative_impl_trait)]
#![feature(const_fn)]
#![feature(core_intrinsics)]
#![feature(decl_macro)]
#![feature(dyn_trait)]
#![feature(fs_read_write)]
#![feature(i128_type)]
#![feature(inclusive_range_syntax)]
#![feature(inclusive_range)]
#![feature(macro_vis_matcher)]
#![feature(match_default_bindings)]
#![feature(never_type)]
#![feature(range_contains)]
#![feature(rustc_diagnostic_macros)]
#![feature(placement_in_syntax)]
#![feature(collection_placement)]
#![feature(nonzero)]
#![feature(underscore_lifetimes)]

#[macro_use]
extern crate bitflags;
#[macro_use] extern crate log;
extern crate graphviz as dot;
#[macro_use]
extern crate rustc;
#[macro_use] extern crate rustc_data_structures;
extern crate serialize as rustc_serialize;
extern crate rustc_errors;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_back;
extern crate rustc_const_math;
extern crate rustc_const_eval;
extern crate core; // for NonZero
extern crate log_settings;
extern crate rustc_apfloat;
extern crate byteorder;
extern crate rustc_trans_utils;

mod diagnostics;

mod borrow_check;
mod build;
mod dataflow;
mod hair;
mod shim;
pub mod transform;
pub mod util;
pub mod interpret;
pub mod monomorphize;

use rustc::ty::maps::Providers;

pub fn provide(providers: &mut Providers) {
    borrow_check::provide(providers);
    shim::provide(providers);
    transform::provide(providers);
    providers.const_eval = interpret::const_eval_provider;
}

#[cfg(not(stage0))] // remove after the next snapshot
__build_diagnostic_array! { librustc_mir, DIAGNOSTICS }
