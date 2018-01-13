// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(rustc_diagnostic_macros)]

#[macro_use]
extern crate rustc;
extern crate rustc_const_eval;
extern crate rustc_const_math;

#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;

use rustc::ty::maps::Providers;

mod diagnostics;

pub mod ast_validation;
pub mod consts;
pub mod hir_stats;
pub mod loops;
mod mir_stats;
pub mod static_recursion;

#[cfg(not(stage0))] // remove after the next snapshot
__build_diagnostic_array! { librustc_passes, DIAGNOSTICS }

pub fn provide(providers: &mut Providers) {
    consts::provide(providers);
}
