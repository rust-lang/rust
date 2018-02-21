// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::traits::{Normalized, ObligationCause};
use rustc::traits::query::NoSolution;
use rustc::ty::{ParamEnvAnd, Ty, TyCtxt};
use rustc::util::common::CellUsizeExt;

crate fn normalize_ty_after_erasing_regions<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    goal: ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Ty<'tcx> {
    let ParamEnvAnd { param_env, value } = goal;
    tcx.sess.perf_stats.normalize_ty_after_erasing_regions.increment();
    tcx.infer_ctxt().enter(|infcx| {
        let cause = ObligationCause::dummy();
        match infcx.at(&cause, param_env).normalize(&value) {
            Ok(Normalized { value: normalized_value, obligations: _ }) => {
                //                                   ^^^^^^^^^^^
                //                   We don't care about the `obligations`,
                //                   they are always only region relations,
                //                   and we are about to erase those anyway.
                let normalized_value = infcx.resolve_type_vars_if_possible(&normalized_value);
                let normalized_value = infcx.tcx.erase_regions(&normalized_value);
                tcx.lift_to_global(&normalized_value).unwrap()
            }
            Err(NoSolution) => bug!("could not fully normalize `{:?}`", value),
        }
    })
}
