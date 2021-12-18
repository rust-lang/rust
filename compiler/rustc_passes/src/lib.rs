//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(crate_visibility_modifier)]
#![feature(map_try_insert)]
#![feature(min_specialization)]
#![feature(nll)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

use rustc_middle::ty::query::Providers;

mod check_attr;
mod check_const;
pub mod dead;
mod diagnostic_items;
pub mod entry;
pub mod hir_id_validator;
pub mod hir_stats;
mod intrinsicck;
mod lang_items;
pub mod layout_test;
mod lib_features;
mod liveness;
pub mod loops;
mod naked_functions;
mod reachable;
mod region;
pub mod stability;
mod upvars;
mod weak_lang_items;

pub fn provide(providers: &mut Providers) {
    check_attr::provide(providers);
    check_const::provide(providers);
    diagnostic_items::provide(providers);
    entry::provide(providers);
    lang_items::provide(providers);
    lib_features::provide(providers);
    loops::provide(providers);
    naked_functions::provide(providers);
    liveness::provide(providers);
    intrinsicck::provide(providers);
    reachable::provide(providers);
    region::provide(providers);
    stability::provide(providers);
    upvars::provide(providers);
}
