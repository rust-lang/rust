//! New recursive solver modeled on Chalk's recursive solver. Most of
//! the guts are broken up into modules; see the comments in those modules.

#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(let_chains)]
#![feature(drain_filter)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod chalk;
mod codegen;
mod dropck_outlives;
mod evaluate_obligation;
mod implied_outlives_bounds;
mod normalize_erasing_regions;
mod normalize_projection_ty;
mod type_op;

pub use type_op::{type_op_ascribe_user_type_with_span, type_op_prove_predicate_with_cause};

use rustc_middle::ty::query::Providers;

pub fn provide(p: &mut Providers) {
    dropck_outlives::provide(p);
    evaluate_obligation::provide(p);
    implied_outlives_bounds::provide(p);
    chalk::provide(p);
    normalize_projection_ty::provide(p);
    normalize_erasing_regions::provide(p);
    type_op::provide(p);
    p.codegen_select_candidate = codegen::codegen_select_candidate;
}
