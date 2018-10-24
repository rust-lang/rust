// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use mir::UserTypeAnnotation;
use traits::query::Fallible;
use ty::{self, ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AscribeUserType<'tcx> {
    pub mir_ty: Ty<'tcx>,
    pub variance: ty::Variance,
    pub user_ty: UserTypeAnnotation<'tcx>,
}

impl<'tcx> AscribeUserType<'tcx> {
    pub fn new(
        mir_ty: Ty<'tcx>,
        variance: ty::Variance,
        user_ty: UserTypeAnnotation<'tcx>,
    ) -> Self {
        AscribeUserType { mir_ty, variance, user_ty }
    }
}

impl<'gcx: 'tcx, 'tcx> super::QueryTypeOp<'gcx, 'tcx> for AscribeUserType<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(
        _tcx: TyCtxt<'_, 'gcx, 'tcx>,
        _key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        None
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, ()>> {
        tcx.type_op_ascribe_user_type(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, ()>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, ()>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for AscribeUserType<'tcx> {
        mir_ty, variance, user_ty
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for AscribeUserType<'a> {
        type Lifted = AscribeUserType<'tcx>;
        mir_ty, variance, user_ty
    }
}

impl_stable_hash_for! {
    struct AscribeUserType<'tcx> {
        mir_ty, variance, user_ty
    }
}
