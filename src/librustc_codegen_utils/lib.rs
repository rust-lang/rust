//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(arbitrary_self_types)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(never_type)]
#![feature(nll)]
#![feature(in_band_lifetimes)]

#![recursion_limit="256"]

#[macro_use]
extern crate rustc;

use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc::hir::def_id::LOCAL_CRATE;
use syntax::symbol::sym;

pub mod link;
pub mod codegen_backend;
pub mod symbol_names;
pub mod symbol_names_test;

/// check for the #[rustc_error] annotation, which forces an
/// error in codegen. This is used to write compile-fail tests
/// that actually test that compilation succeeds without
/// reporting an error.
pub fn check_for_rustc_errors_attr(tcx: TyCtxt<'_>) {
    if let Some((def_id, _)) = tcx.entry_fn(LOCAL_CRATE) {
        if tcx.has_attr(def_id, sym::rustc_error) {
            tcx.sess.span_fatal(tcx.def_span(def_id), "compilation successful");
        }
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    crate::symbol_names::provide(providers);
}
