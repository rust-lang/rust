// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! constant evaluation on the HIR and code to validate patterns/matches
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(rustc_diagnostic_macros)]
#![feature(slice_patterns)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(i128_type)]
#![feature(from_ref)]

extern crate arena;
#[macro_use] extern crate syntax;
#[macro_use] extern crate log;
#[macro_use] extern crate rustc;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate syntax_pos;

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
mod diagnostics;

mod eval;
mod _match;
pub mod check_match;
pub mod pattern;

pub use eval::*;

use rustc::ty::maps::Providers;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        check_match: check_match::check_match,
        ..*providers
    };
}

// Build the diagnostics array at the end so that the metadata includes error use sites.
#[cfg(not(stage0))] // remove after the next snapshot
__build_diagnostic_array! { librustc_const_eval, DIAGNOSTICS }
