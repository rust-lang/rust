//! New recursive solver modeled on Chalk's recursive solver. Most of
//! the guts are broken up into modules; see the comments in those modules.

#![feature(crate_visibility_modifier)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![recursion_limit = "256"]

extern crate chalk_engine;
#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_target;
extern crate smallvec;
extern crate syntax;
extern crate syntax_pos;

mod chalk_context;
mod dropck_outlives;
mod evaluate_obligation;
mod generic_types;
mod implied_outlives_bounds;
pub mod lowering;
mod normalize_erasing_regions;
mod normalize_projection_ty;
mod type_op;

use rustc::ty::query::Providers;

pub fn provide(p: &mut Providers) {
    dropck_outlives::provide(p);
    evaluate_obligation::provide(p);
    implied_outlives_bounds::provide(p);
    lowering::provide(p);
    chalk_context::provide(p);
    normalize_projection_ty::provide(p);
    normalize_erasing_regions::provide(p);
    type_op::provide(p);
}
