//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(in_band_lifetimes)]
#![feature(nll)]

#![recursion_limit="256"]

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;

use rustc::ty::query::Providers;

pub mod ast_validation;
mod check_const;
pub mod hir_stats;
pub mod layout_test;
pub mod loops;
pub mod dead;
pub mod entry;
mod liveness;
mod intrinsicck;

pub fn provide(providers: &mut Providers<'_>) {
    check_const::provide(providers);
    entry::provide(providers);
    loops::provide(providers);
    liveness::provide(providers);
    intrinsicck::provide(providers);
}
