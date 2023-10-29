//! Queries that are independent from the main solver code.

#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(let_chains)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod codegen;
mod dropck_outlives;
mod evaluate_obligation;
mod implied_outlives_bounds;
mod normalize_erasing_regions;
mod normalize_projection_ty;
mod type_op;

pub use rustc_trait_selection::traits::query::type_op::ascribe_user_type::type_op_ascribe_user_type_with_span;
pub use type_op::type_op_prove_predicate_with_cause;

use rustc_middle::query::Providers;

pub fn provide(providers: &mut Providers) {
    dropck_outlives::provide(providers);
    evaluate_obligation::provide(providers);
    implied_outlives_bounds::provide(providers);
    normalize_projection_ty::provide(providers);
    normalize_erasing_regions::provide(providers);
    type_op::provide(providers);
    query_provider!(
        providers,
        provide(codegen_select_candidate) = codegen::codegen_select_candidate
    );
}
