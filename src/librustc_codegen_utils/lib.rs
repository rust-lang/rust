// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(custom_attribute)]
#![feature(nll)]
#![allow(unused_attributes)]
#![allow(dead_code)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

extern crate flate2;
#[macro_use]
extern crate log;

#[macro_use]
extern crate rustc;
extern crate rustc_target;
extern crate rustc_mir;
extern crate rustc_incremental;
extern crate syntax;
extern crate syntax_pos;
#[macro_use] extern crate rustc_data_structures;
extern crate rustc_metadata_utils;

use rustc::ty::TyCtxt;

pub mod link;
pub mod codegen_backend;
pub mod symbol_names;
pub mod symbol_names_test;

/// check for the #[rustc_error] annotation, which forces an
/// error in codegen. This is used to write compile-fail tests
/// that actually test that compilation succeeds without
/// reporting an error.
pub fn check_for_rustc_errors_attr(tcx: TyCtxt) {
    if let Some((id, span, _)) = *tcx.sess.entry_fn.borrow() {
        let main_def_id = tcx.hir.local_def_id(id);

        if tcx.has_attr(main_def_id, "rustc_error") {
            tcx.sess.span_fatal(span, "compilation successful");
        }
    }
}

__build_diagnostic_array! { librustc_codegen_utils, DIAGNOSTICS }
