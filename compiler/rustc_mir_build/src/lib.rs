//! Construction of MIR from HIR.
//!
//! This crate also contains the match exhaustiveness and usefulness checking.
#![allow(rustc::potential_query_instability)]
#![feature(assert_matches)]
#![feature(associated_type_bounds)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod build;
mod check_unsafety;
mod errors;
pub mod lints;
mod thir;

use rustc_middle::query::Providers;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;

fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    query_provider!(
        providers,
        provide(check_match) = thir::pattern::check_match,
        provide(lit_to_const) = thir::constant::lit_to_const,
        provide(mir_built) = build::mir_built,
        provide(closure_saved_names_of_captured_variables) =
            build::closure_saved_names_of_captured_variables,
        provide(thir_check_unsafety) = check_unsafety::thir_check_unsafety,
        provide(thir_body) = thir::cx::thir_body,
        provide(thir_tree) = thir::print::thir_tree,
        provide(thir_flat) = thir::print::thir_flat,
    );
}
