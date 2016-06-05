// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Module for inferring the variance of type and lifetime
//! parameters. See README.md for details.

use arena;
use rustc::ty::TyCtxt;

/// Defines the `TermsContext` basically houses an arena where we can
/// allocate terms.
mod terms;

/// Code to gather up constraints.
mod constraints;

/// Code to solve constraints and write out the results.
mod solve;

/// Code for transforming variances.
mod xform;

pub fn infer_variance<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut arena = arena::TypedArena::new();
    let terms_cx = terms::determine_parameters_to_be_inferred(tcx, &mut arena);
    let constraints_cx = constraints::add_constraints_from_crate(terms_cx);
    solve::solve_constraints(constraints_cx);
    tcx.variance_computed.set(true);
}
