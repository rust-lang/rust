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
use traits::query::Fallible;
use hir::def_id::DefId;
use mir::ProjectionKind;
use ty::{self, ParamEnvAnd, Ty, TyCtxt};
use ty::subst::UserSubsts;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AscribeUserType<'tcx> {
    pub mir_ty: Ty<'tcx>,
    pub variance: ty::Variance,
    pub def_id: DefId,
    pub user_substs: UserSubsts<'tcx>,
    pub projs: &'tcx ty::List<ProjectionKind<'tcx>>,
}

impl<'tcx> AscribeUserType<'tcx> {
    pub fn new(
        mir_ty: Ty<'tcx>,
        variance: ty::Variance,
        def_id: DefId,
        user_substs: UserSubsts<'tcx>,
        projs: &'tcx ty::List<ProjectionKind<'tcx>>,
    ) -> Self {
        AscribeUserType { mir_ty, variance, def_id, user_substs, projs }
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
        mir_ty, variance, def_id, user_substs, projs
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for AscribeUserType<'a> {
        type Lifted = AscribeUserType<'tcx>;
        mir_ty, variance, def_id, user_substs, projs
    }
}

impl_stable_hash_for! {
    struct AscribeUserType<'tcx> {
        mir_ty, variance, def_id, user_substs, projs
    }
}
