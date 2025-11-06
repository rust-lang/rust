//! Various checks
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![feature(if_let_guard)]
#![feature(map_try_insert)]
// tidy-alphabetical-end

use rustc_middle::util::Providers;

pub mod abi_test;
mod check_attr;
mod check_export;
pub mod dead;
mod debugger_visualizer;
mod diagnostic_items;
pub mod entry;
mod errors;
pub mod hir_id_validator;
pub mod input_stats;
mod lang_items;
pub mod layout_test;
mod lib_features;
mod reachable;
pub mod stability;
mod upvars;
mod weak_lang_items;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    check_attr::provide(providers);
    dead::provide(providers);
    debugger_visualizer::provide(providers);
    diagnostic_items::provide(providers);
    entry::provide(providers);
    lang_items::provide(providers);
    lib_features::provide(providers);
    reachable::provide(providers);
    stability::provide(providers);
    upvars::provide(providers);
    check_export::provide(providers);
}
