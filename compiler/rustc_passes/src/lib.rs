//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(try_blocks)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

use rustc_middle::query::Providers;

pub mod abi_test;
mod check_attr;
mod check_const;
pub mod dead;
mod debugger_visualizer;
mod diagnostic_items;
pub mod entry;
mod errors;
pub mod hir_id_validator;
pub mod hir_stats;
mod lang_items;
pub mod layout_test;
mod lib_features;
mod liveness;
pub mod loops;
mod naked_functions;
mod reachable;
pub mod stability;
mod upvars;
mod weak_lang_items;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    check_attr::provide(providers);
    check_const::provide(providers);
    dead::provide(providers);
    debugger_visualizer::provide(providers);
    diagnostic_items::provide(providers);
    entry::provide(providers);
    lang_items::provide(providers);
    lib_features::provide(providers);
    loops::provide(providers);
    naked_functions::provide(providers);
    liveness::provide(providers);
    reachable::provide(providers);
    stability::provide(providers);
    upvars::provide(providers);
}
