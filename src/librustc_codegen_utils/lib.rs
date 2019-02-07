//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(custom_attribute)]
#![feature(nll)]
#![allow(unused_attributes)]
#![feature(rustc_diagnostic_macros)]
#![feature(in_band_lifetimes)]

#![recursion_limit="256"]

extern crate flate2;
#[macro_use]
extern crate log;

#[macro_use]
extern crate rustc;
extern crate rustc_target;
extern crate rustc_metadata;
extern crate rustc_mir;
extern crate rustc_incremental;
extern crate syntax;
extern crate syntax_pos;
#[macro_use] extern crate rustc_data_structures;

use rustc::ty::TyCtxt;
use rustc::hir::def_id::LOCAL_CRATE;

pub mod link;
pub mod codegen_backend;
pub mod symbol_names;
pub mod symbol_names_test;

/// check for the #[rustc_error] annotation, which forces an
/// error in codegen. This is used to write compile-fail tests
/// that actually test that compilation succeeds without
/// reporting an error.
pub fn check_for_rustc_errors_attr(tcx: TyCtxt) {
    if let Some((def_id, _)) = tcx.entry_fn(LOCAL_CRATE) {
        if tcx.has_attr(def_id, "rustc_error") {
            tcx.sess.span_fatal(tcx.def_span(def_id), "compilation successful");
        }
    }
}
