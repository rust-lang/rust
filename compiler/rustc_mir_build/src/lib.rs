//! Construction of MIR from HIR.
//!
//! This crate also contains the match exhaustiveness and usefulness checking.
#![allow(rustc::potential_query_instability)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(once_cell)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod build;
mod check_unsafety;
mod lints;
pub mod thir;

use rustc_middle::ty::query::Providers;

pub fn provide(providers: &mut Providers) {
    providers.check_match = thir::pattern::check_match;
    providers.lit_to_const = thir::constant::lit_to_const;
    providers.lit_to_mir_constant = build::lit_to_mir_constant;
    providers.mir_built = build::mir_built;
    providers.thir_check_unsafety = check_unsafety::thir_check_unsafety;
    providers.thir_check_unsafety_for_const_arg = check_unsafety::thir_check_unsafety_for_const_arg;
    providers.thir_body = thir::cx::thir_body;
    providers.thir_tree = thir::cx::thir_tree;
}
