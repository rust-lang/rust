// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "rustc_borrowck"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![allow(non_camel_case_types)]

#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(into_cow)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;

// for "clarity", rename the graphviz crate to dot; graphviz within `borrowck`
// refers to the borrowck-specific graphviz adapter traits.
extern crate graphviz as dot;
extern crate rustc;
extern crate rustc_front;

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;
pub use borrowck::FnPartsWithCFG;

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostics;

mod borrowck;

pub mod graphviz;

__build_diagnostic_array! { librustc_borrowck, DIAGNOSTICS }
