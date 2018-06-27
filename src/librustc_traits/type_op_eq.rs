// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::canonical::{Canonical, QueryResult};
use rustc::traits::query::type_op::eq::Eq;
use rustc::traits::query::NoSolution;
use rustc::traits::ObligationCause;
use rustc::ty::TyCtxt;
use rustc_data_structures::sync::Lrc;

crate fn type_op_eq<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Eq<'tcx>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, Eq { param_env, a, b }| {
            Ok(infcx.at(&ObligationCause::dummy(), param_env).eq(a, b)?)
        })
}
