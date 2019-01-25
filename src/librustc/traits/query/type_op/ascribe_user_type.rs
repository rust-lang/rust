use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use traits::query::Fallible;
use hir::def_id::DefId;
use ty::{ParamEnvAnd, Ty, TyCtxt};
use ty::subst::UserSubsts;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AscribeUserType<'tcx> {
    pub mir_ty: Ty<'tcx>,
    pub def_id: DefId,
    pub user_substs: UserSubsts<'tcx>,
}

impl<'tcx> AscribeUserType<'tcx> {
    pub fn new(
        mir_ty: Ty<'tcx>,
        def_id: DefId,
        user_substs: UserSubsts<'tcx>,
    ) -> Self {
        Self { mir_ty,  def_id, user_substs }
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
        mir_ty, def_id, user_substs
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for AscribeUserType<'a> {
        type Lifted = AscribeUserType<'tcx>;
        mir_ty, def_id, user_substs
    }
}

impl_stable_hash_for! {
    struct AscribeUserType<'tcx> {
        mir_ty, def_id, user_substs
    }
}
