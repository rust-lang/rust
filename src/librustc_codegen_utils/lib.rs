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

#![deny(rust_2018_idioms)]

#[macro_use]
extern crate rustc;
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
pub fn check_for_rustc_errors_attr(tcx: TyCtxt<'_, '_, '_>) {
    if let Some((def_id, _)) = tcx.entry_fn(LOCAL_CRATE) {
        if tcx.has_attr(def_id, "rustc_error") {
            tcx.sess.span_fatal(tcx.def_span(def_id), "compilation successful");
        }
    }
}
