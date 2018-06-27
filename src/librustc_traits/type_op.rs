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
use rustc::infer::{InferCtxt, InferOk};
use rustc::traits::query::type_op::eq::Eq;
use rustc::traits::query::type_op::normalize::Normalize;
use rustc::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc::traits::query::type_op::subtype::Subtype;
use rustc::traits::query::{Fallible, NoSolution};
use rustc::traits::{Obligation, Normalized, ObligationCause};
use rustc::ty::query::Providers;
use rustc::ty::{FnSig, Lift, PolyFnSig, Predicate, Ty, TyCtxt, TypeFoldable};
use rustc_data_structures::sync::Lrc;
use std::fmt;

crate fn provide(p: &mut Providers) {
    *p = Providers {
        type_op_eq,
        type_op_prove_predicate,
        type_op_subtype,
        type_op_normalize_ty,
        type_op_normalize_predicate,
        type_op_normalize_fn_sig,
        type_op_normalize_poly_fn_sig,
        ..*p
    };
}

fn type_op_eq<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Eq<'tcx>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, Eq { param_env, a, b }| {
            Ok(infcx.at(&ObligationCause::dummy(), param_env).eq(a, b)?)
        })
}

fn type_op_normalize<T>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    key: Normalize<'tcx, T>,
) -> Fallible<InferOk<'tcx, T>>
where
    T: fmt::Debug + TypeFoldable<'tcx> + Lift<'gcx>,
{
    let Normalize { param_env, value } = key;
    let Normalized { value, obligations } = infcx
        .at(&ObligationCause::dummy(), param_env)
        .normalize(&value)?;
    Ok(InferOk { value, obligations }) // ugh we should merge these two structs
}

fn type_op_normalize_ty(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, Ty<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, Ty<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_predicate(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, Predicate<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, Predicate<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_fn_sig(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, FnSig<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, FnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_poly_fn_sig(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, PolyFnSig<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, PolyFnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_subtype<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Subtype<'tcx>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(
        &canonicalized,
        |infcx,
         Subtype {
             param_env,
             sub,
             sup,
         }| {
            Ok(infcx
                .at(&ObligationCause::dummy(), param_env)
                .sup(sup, sub)?)
        },
    )
}

fn type_op_prove_predicate<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ProvePredicate<'tcx>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |_infcx, key| {
            let ProvePredicate {
                param_env,
                predicate,
            } = key;
            Ok(InferOk {
                value: (),
                obligations: vec![Obligation::new(
                    ObligationCause::dummy(),
                    param_env,
                    predicate,
                )],
            })
        })
}
