//! New recursive solver modeled on Chalk's recursive solver. Most of
//! the guts are broken up into modules; see the comments in those modules.

#![feature(crate_visibility_modifier)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(control_flow_enum)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

mod chalk;
mod dropck_outlives;
mod evaluate_obligation;
mod implied_outlives_bounds;
mod normalize_erasing_regions;
mod normalize_projection_ty;
mod type_op;

use rustc_middle::ty::query::Providers;

pub fn provide(p: &mut Providers) {
    dropck_outlives::provide(p);
    evaluate_obligation::provide(p);
    implied_outlives_bounds::provide(p);
    chalk::provide(p);
    normalize_projection_ty::provide(p);
    normalize_erasing_regions::provide(p);
    type_op::provide(p);
}
