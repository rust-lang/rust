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

#![crate_name = "rustc_passes"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(rustc_diagnostic_macros)]
#![feature(staged_api)]
#![feature(rustc_private)]

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

pub mod diagnostics;

pub mod ast_validation;
pub mod consts;
pub mod hir_stats;
pub mod loops;
pub mod mir_stats;
pub mod no_asm;
pub mod rvalues;
pub mod static_recursion;
