use crate::infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use std::fmt;
use crate::traits::query::Fallible;
use crate::ty::fold::TypeFoldable;
use crate::ty::{self, Lift, ParamEnvAnd, Ty, TyCtxt};

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

impl<'tcx, T> super::QueryTypeOp<'tcx> for Normalize<T>
where
    T: Normalizable<'tcx> + 'tcx,
{
    type QueryResponse = T;

    fn try_fast_path(_tcx: TyCtxt<'tcx>, key: &ParamEnvAnd<'tcx, Self>) -> Option<T> {
        if !key.value.value.has_projections() {
            Some(key.value.value)
        } else {
            None
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self::QueryResponse>> {
        T::type_op_method(tcx, canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, T>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, T>> {
        T::shrink_to_tcx_lifetime(v)
    }
}

pub trait Normalizable<'tcx>: fmt::Debug + TypeFoldable<'tcx> + Lift<'tcx> + Copy {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self>>;

    /// Converts from the `'tcx` (lifted) form of `Self` into the `tcx`
    /// form of `Self`.
    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>>;
}

impl Normalizable<'tcx> for Ty<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self>> {
        tcx.type_op_normalize_ty(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'tcx> for ty::Predicate<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self>> {
        tcx.type_op_normalize_predicate(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'tcx> for ty::PolyFnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self>> {
        tcx.type_op_normalize_poly_fn_sig(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, Self>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, Self>> {
        v
    }
}

impl Normalizable<'tcx> for ty::FnSig<'tcx> {
    fn type_op_method(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Normalize<Self>>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self>> {
        tcx.type_op_normalize_fn_sig(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, Self>,
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
    impl<T> for struct Normalize<T> {
        value
    }
}
