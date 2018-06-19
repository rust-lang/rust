// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResult, QueryResult};
use std::fmt;
use traits::query::Fallible;
use ty::fold::TypeFoldable;
use ty::{self, Lift, ParamEnv, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Normalize<'tcx, T> {
    pub param_env: ParamEnv<'tcx>,
    pub value: T,
}

impl<'tcx, T> Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    pub fn new(param_env: ParamEnv<'tcx>, value: T) -> Self {
        Self { param_env, value }
    }
}

impl<'gcx: 'tcx, 'tcx, T> super::QueryTypeOp<'gcx, 'tcx> for Normalize<'tcx, T>
where
    T: Normalizable<'gcx, 'tcx>,
{
    type QueryKey = Self;
    type QueryResult = T;

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<T, Self> {
        if !self.value.has_projections() {
            Ok(self.value)
        } else {
            Err(self)
        }
    }

    fn into_query_key(self) -> Self {
        self
    }

    fn param_env(&self) -> ParamEnv<'tcx> {
        self.param_env
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Self>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self::QueryResult>> {
        T::type_op_method(tcx, canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, T>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, T>> {
        T::upcast_result(v)
    }
}

pub trait Normalizable<'gcx, 'tcx>: fmt::Debug + TypeFoldable<'tcx> + Lift<'gcx> {
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Normalize<'gcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self>>;

    /// Convert from the `'gcx` (lifted) form of `Self` into the `tcx`
    /// form of `Self`.
    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self>>;
}

impl Normalizable<'gcx, 'tcx> for Ty<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Normalize<'gcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self>> {
        tcx.type_op_normalize_ty(canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::Predicate<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Normalize<'gcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self>> {
        tcx.type_op_normalize_predicate(canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::PolyFnSig<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Normalize<'gcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self>> {
        tcx.type_op_normalize_poly_fn_sig(canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::FnSig<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Normalize<'gcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self>> {
        tcx.type_op_normalize_fn_sig(canonicalized)
    }

    fn upcast_result(
        v: &'a CanonicalizedQueryResult<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Normalize<'tcx, T> {
        param_env,
        value,
    } where T: TypeFoldable<'tcx>,
}

BraceStructLiftImpl! {
    impl<'a, 'tcx, T> Lift<'tcx> for Normalize<'a, T> {
        type Lifted = Normalize<'tcx, T::Lifted>;
        param_env,
        value,
    } where T: Lift<'tcx>,
}

impl_stable_hash_for! {
    impl<'tcx, T> for struct Normalize<'tcx, T> {
        param_env, value
    }
}
