//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(
    html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
    html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
    html_root_url = "https://doc.rust-lang.org/nightly/"
)]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;

#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate rustc_errors as errors;
extern crate syntax_pos;

use rustc::ty::query::Providers;

mod diagnostics;

pub mod ast_validation;
pub mod hir_stats;
pub mod loops;
pub mod rvalue_promotion;

__build_diagnostic_array! { librustc_passes, DIAGNOSTICS }

pub fn provide(providers: &mut Providers) {
    rvalue_promotion::provide(providers);
}
