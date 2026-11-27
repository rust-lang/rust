//! Queries that are independent from the main solver code.

// tidy-alphabetical-start
#![recursion_limit = "256"]
// tidy-alphabetical-end

mod codegen;
mod coroutine_witnesses;
mod dropck_outlives;
mod evaluate_obligation;
mod implied_outlives_bounds;
mod normalize_erasing_regions;
mod normalize_projection_ty;
mod type_op;

use rustc_middle::query::Providers;

pub fn provide(p: &mut Providers) {
    dropck_outlives::provide(p);
    evaluate_obligation::provide(p);
    implied_outlives_bounds::provide(p);
    normalize_projection_ty::provide(p);
    normalize_erasing_regions::provide(p);
    type_op::provide(p);
    p.codegen_select_candidate = codegen::codegen_select_candidate;
    p.coroutine_hidden_types = coroutine_witnesses::coroutine_hidden_types;
}
