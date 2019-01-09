use infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use std::fmt;
use traits::query::Fallible;
use ty::fold::TypeFoldable;
use ty::{self, Lift, ParamEnvAnd, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Normalize<T> {
    pub value: T,
}

impl<'tcx, T> Normalize<T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<'gcx: 'tcx, 'tcx, T> super::QueryTypeOp<'gcx, 'tcx> for Normalize<T>
where
    T: Normalizable<'gcx, 'tcx>,
{
    type QueryResponse = T;

    fn try_fast_path(_tcx: TyCtxt<'_, 'gcx, 'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<T> {
        if !key.value.value.has_projections() {
            Some(key.value.value)
        } else {
            None
        }
    }

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self::QueryResponse>> {
        T::type_op_method(tcx, canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, T>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, T>> {
        T::shrink_to_tcx_lifetime(v)
    }
}

pub trait Normalizable<'gcx, 'tcx>: fmt::Debug + TypeFoldable<'tcx> + Lift<'gcx> + Copy {
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self>>;

    /// Convert from the `'gcx` (lifted) form of `Self` into the `tcx`
    /// form of `Self`.
    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>>;
}

impl Normalizable<'gcx, 'tcx> for Ty<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self>> {
        tcx.type_op_normalize_ty(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::Predicate<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self>> {
        tcx.type_op_normalize_predicate(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::PolyFnSig<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self>> {
        tcx.type_op_normalize_poly_fn_sig(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'gcx, 'tcx> for ty::FnSig<'tcx>
where
    'gcx: 'tcx,
{
    fn type_op_method(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'gcx, Self>> {
        tcx.type_op_normalize_fn_sig(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'gcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Normalize<T> {
        value,
    } where T: TypeFoldable<'tcx>,
}

BraceStructLiftImpl! {
    impl<'tcx, T> Lift<'tcx> for Normalize<T> {
        type Lifted = Normalize<T::Lifted>;
        value,
    } where T: Lift<'tcx>,
}

impl_stable_hash_for! {
    impl<'tcx, T> for struct Normalize<T> {
        value
    }
}
