// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_borrowck"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![allow(non_camel_case_types)]

#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(associated_consts)]
#![feature(nonzero)]
#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;

// for "clarity", rename the graphviz crate to dot; graphviz within `borrowck`
// refers to the borrowck-specific graphviz adapter traits.
extern crate graphviz as dot;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate core; // for NonZero

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;
pub use borrowck::{AnalysisData, BorrowckCtxt, ElaborateDrops};

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostics;

mod borrowck;

pub mod graphviz;

__build_diagnostic_array! { librustc_borrowck, DIAGNOSTICS }
