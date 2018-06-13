// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! New recursive solver modeled on Chalk's recursive solver. Most of
//! the guts are broken up into modules; see the comments in those modules.

#![feature(crate_in_paths)]
#![feature(crate_visibility_modifier)]
#![feature(extern_prelude)]
#![feature(iterator_find_map)]
#![feature(in_band_lifetimes)]

#![recursion_limit="256"]

extern crate chalk_engine;
#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;
extern crate syntax;
extern crate syntax_pos;

mod chalk_context;
mod dropck_outlives;
mod evaluate_obligation;
mod normalize_projection_ty;
mod normalize_erasing_regions;
mod util;
pub mod lowering;

use rustc::ty::query::Providers;

pub fn provide(p: &mut Providers) {
    *p = Providers {
        dropck_outlives: dropck_outlives::dropck_outlives,
        adt_dtorck_constraint: dropck_outlives::adt_dtorck_constraint,
        normalize_projection_ty: normalize_projection_ty::normalize_projection_ty,
        normalize_ty_after_erasing_regions:
            normalize_erasing_regions::normalize_ty_after_erasing_regions,
        program_clauses_for: lowering::program_clauses_for,
        program_clauses_for_env: lowering::program_clauses_for_env,
        evaluate_obligation: evaluate_obligation::evaluate_obligation,
        ..*p
    };
}
