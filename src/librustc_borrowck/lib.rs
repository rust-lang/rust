// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![allow(non_camel_case_types)]

#![feature(from_ref)]
#![feature(match_default_bindings)]
#![feature(quote)]

#[macro_use] extern crate log;
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;

// for "clarity", rename the graphviz crate to dot; graphviz within `borrowck`
// refers to the borrowck-specific graphviz adapter traits.
extern crate graphviz as dot;
#[macro_use]
extern crate rustc;
extern crate rustc_mir;

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;

mod borrowck;

pub mod graphviz;

pub use borrowck::provide;
