//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(slice_patterns)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;

use rustc::ty::query::Providers;

pub mod ast_validation;
mod check_const;
pub mod dead;
mod diagnostic_items;
pub mod entry;
pub mod hir_stats;
mod intrinsicck;
pub mod layout_test;
mod lib_features;
mod liveness;
pub mod loops;
mod reachable;
mod region;
pub mod stability;

pub fn provide(providers: &mut Providers<'_>) {
    check_const::provide(providers);
    diagnostic_items::provide(providers);
    entry::provide(providers);
    lib_features::provide(providers);
    loops::provide(providers);
    liveness::provide(providers);
    intrinsicck::provide(providers);
    reachable::provide(providers);
    region::provide(providers);
    stability::provide(providers);
}
