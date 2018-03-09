// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Experimental types for the trait query interface. The methods
//! defined in this module are all based on **canonicalization**,
//! which makes a canonical query by replacing unbound inference
//! variables and regions, so that results can be reused more broadly.
//! The providers for the queries defined here can be found in
//! `librustc_traits`.

use infer::canonical::Canonical;
use ty::{self, Ty};

pub mod dropck_outlives;
pub mod evaluate_obligation;
pub mod normalize;
pub mod normalize_erasing_regions;

pub type CanonicalProjectionGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::ProjectionTy<'tcx>>>;

pub type CanonicalTyGoal<'tcx> = Canonical<'tcx, ty::ParamEnvAnd<'tcx, Ty<'tcx>>>;

pub type CanonicalPredicateGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::Predicate<'tcx>>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NoSolution;

pub type Fallible<T> = Result<T, NoSolution>;

impl_stable_hash_for!(struct NoSolution { });
