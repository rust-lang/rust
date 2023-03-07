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
#![feature(once_cell)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod build;
mod check_unsafety;
mod errors;
mod lints;
pub mod thir;

use rustc_middle::ty::query::Providers;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_macros::fluent_messages;

fluent_messages! { "../locales/en-US.ftl" }

pub fn provide(providers: &mut Providers) {
    providers.check_match = thir::pattern::check_match;
    providers.lit_to_const = thir::constant::lit_to_const;
    providers.lit_to_mir_constant = build::lit_to_mir_constant;
    providers.mir_built = build::mir_built;
    providers.thir_check_unsafety = check_unsafety::thir_check_unsafety;
    providers.thir_check_unsafety_for_const_arg = check_unsafety::thir_check_unsafety_for_const_arg;
    providers.thir_body = thir::cx::thir_body;
    providers.thir_tree = thir::print::thir_tree;
    providers.thir_flat = thir::print::thir_flat;
}
