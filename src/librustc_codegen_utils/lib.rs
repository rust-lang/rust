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
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc;

use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use syntax::symbol::sym;

pub mod codegen_backend;
pub mod link;
pub mod symbol_names;
pub mod symbol_names_test;

pub fn trigger_delay_span_bug(tcx: TyCtxt<'_>, key: DefId) {
    tcx.sess.delay_span_bug(
        tcx.def_span(key),
        "delayed span bug triggered by #[rustc_error(delay_span_bug_from_inside_query)]",
    );
}

/// check for the #[rustc_error] annotation, which forces an
/// error in codegen. This is used to write compile-fail tests
/// that actually test that compilation succeeds without
/// reporting an error.
pub fn check_for_rustc_errors_attr(tcx: TyCtxt<'_>) {
    if let Some((def_id, _)) = tcx.entry_fn(LOCAL_CRATE) {
        let attrs = &*tcx.get_attrs(def_id);
        for attr in attrs {
            if attr.check_name(sym::rustc_error) {
                match attr.meta_item_list() {
                    // check if there is a #[rustc_error(delayed)]
                    Some(list) => {
                        if list.iter().any(|list_item| {
                            list_item.ident().map(|i| i.name)
                                == Some(sym::delay_span_bug_from_inside_query)
                        }) {
                            tcx.ensure().trigger_delay_span_bug(def_id);
                        }
                    }
                    // bare #[rustc_error]
                    None => {
                        tcx.sess.span_fatal(
                            tcx.def_span(def_id),
                            "fatal error triggered by #[rustc_error]",
                        );
                    }
                }
            }
        }
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    crate::symbol_names::provide(providers);
    *providers = Providers { trigger_delay_span_bug, ..*providers };
}
