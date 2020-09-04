//! Experimental types for the trait query interface. The methods
//! defined in this module are all based on **canonicalization**,
//! which makes a canonical query by replacing unbound inference
//! variables and regions, so that results can be reused more broadly.
//! The providers for the queries defined here can be found in
//! `librustc_traits`.

use crate::ich::StableHashingContext;
use crate::infer::canonical::{Canonical, QueryResponse};
use crate::ty::error::TypeError;
use crate::ty::subst::GenericArg;
use crate::ty::{self, Ty, TyCtxt};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_errors::struct_span_err;
use rustc_span::source_map::Span;
use std::iter::FromIterator;
use std::mem;

pub mod type_op {
    use crate::ty::fold::TypeFoldable;
    use crate::ty::subst::UserSubsts;
    use crate::ty::{Predicate, Ty};
    use rustc_hir::def_id::DefId;
    use std::fmt;

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
    pub struct AscribeUserType<'tcx> {
        pub mir_ty: Ty<'tcx>,
        pub def_id: DefId,
        pub user_substs: UserSubsts<'tcx>,
    }

    impl<'tcx> AscribeUserType<'tcx> {
        pub fn new(mir_ty: Ty<'tcx>, def_id: DefId, user_substs: UserSubsts<'tcx>) -> Self {
            Self { mir_ty, def_id, user_substs }
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
    pub struct Eq<'tcx> {
        pub a: Ty<'tcx>,
        pub b: Ty<'tcx>,
    }

    impl<'tcx> Eq<'tcx> {
        pub fn new(a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
            Self { a, b }
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
    pub struct Subtype<'tcx> {
        pub sub: Ty<'tcx>,
        pub sup: Ty<'tcx>,
    }

    impl<'tcx> Subtype<'tcx> {
        pub fn new(sub: Ty<'tcx>, sup: Ty<'tcx>) -> Self {
            Self { sub, sup }
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
    pub struct ProvePredicate<'tcx> {
        pub predicate: Predicate<'tcx>,
    }

    impl<'tcx> ProvePredicate<'tcx> {
        pub fn new(predicate: Predicate<'tcx>) -> Self {
            ProvePredicate { predicate }
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, HashStable, TypeFoldable, Lift)]
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
}

pub type CanonicalProjectionGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::ProjectionTy<'tcx>>>;

pub type CanonicalTyGoal<'tcx> = Canonical<'tcx, ty::ParamEnvAnd<'tcx, Ty<'tcx>>>;

pub type CanonicalPredicateGoal<'tcx> = Canonical<'tcx, ty::ParamEnvAnd<'tcx, ty::Predicate<'tcx>>>;

pub type CanonicalTypeOpAscribeUserTypeGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::AscribeUserType<'tcx>>>;

pub type CanonicalTypeOpEqGoal<'tcx> = Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Eq<'tcx>>>;

pub type CanonicalTypeOpSubtypeGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Subtype<'tcx>>>;

pub type CanonicalTypeOpProvePredicateGoal<'tcx> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::ProvePredicate<'tcx>>>;

pub type CanonicalTypeOpNormalizeGoal<'tcx, T> =
    Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Normalize<T>>>;

#[derive(Clone, Debug, HashStable)]
pub struct NoSolution;

pub type Fallible<T> = Result<T, NoSolution>;

impl<'tcx> From<TypeError<'tcx>> for NoSolution {
    fn from(_: TypeError<'tcx>) -> NoSolution {
        NoSolution
    }
}

#[derive(Clone, Debug, Default, HashStable, TypeFoldable, Lift)]
pub struct DropckOutlivesResult<'tcx> {
    pub kinds: Vec<GenericArg<'tcx>>,
    pub overflows: Vec<Ty<'tcx>>,
}

impl<'tcx> DropckOutlivesResult<'tcx> {
    pub fn report_overflows(&self, tcx: TyCtxt<'tcx>, span: Span, ty: Ty<'tcx>) {
        if let Some(overflow_ty) = self.overflows.get(0) {
            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0320,
                "overflow while adding drop-check rules for {}",
                ty,
            );
            err.note(&format!("overflowed on {}", overflow_ty));
            err.emit();
        }
    }

    pub fn into_kinds_reporting_overflows(
        self,
        tcx: TyCtxt<'tcx>,
        span: Span,
        ty: Ty<'tcx>,
    ) -> Vec<GenericArg<'tcx>> {
        self.report_overflows(tcx, span, ty);
        let DropckOutlivesResult { kinds, overflows: _ } = self;
        kinds
    }
}

/// A set of constraints that need to be satisfied in order for
/// a type to be valid for destruction.
#[derive(Clone, Debug, HashStable)]
pub struct DtorckConstraint<'tcx> {
    /// Types that are required to be alive in order for this
    /// type to be valid for destruction.
    pub outlives: Vec<ty::subst::GenericArg<'tcx>>,

    /// Types that could not be resolved: projections and params.
    pub dtorck_types: Vec<Ty<'tcx>>,

    /// If, during the computation of the dtorck constraint, we
    /// overflow, that gets recorded here. The caller is expected to
    /// report an error.
    pub overflows: Vec<Ty<'tcx>>,
}

impl<'tcx> DtorckConstraint<'tcx> {
    pub fn empty() -> DtorckConstraint<'tcx> {
        DtorckConstraint { outlives: vec![], dtorck_types: vec![], overflows: vec![] }
    }
}

impl<'tcx> FromIterator<DtorckConstraint<'tcx>> for DtorckConstraint<'tcx> {
    fn from_iter<I: IntoIterator<Item = DtorckConstraint<'tcx>>>(iter: I) -> Self {
        let mut result = Self::empty();

        for DtorckConstraint { outlives, dtorck_types, overflows } in iter {
            result.outlives.extend(outlives);
            result.dtorck_types.extend(dtorck_types);
            result.overflows.extend(overflows);
        }

        result
    }
}

/// This returns true if the type `ty` is "trivial" for
/// dropck-outlives -- that is, if it doesn't require any types to
/// outlive. This is similar but not *quite* the same as the
/// `needs_drop` test in the compiler already -- that is, for every
/// type T for which this function return true, needs-drop would
/// return `false`. But the reverse does not hold: in particular,
/// `needs_drop` returns false for `PhantomData`, but it is not
/// trivial for dropck-outlives.
///
/// Note also that `needs_drop` requires a "global" type (i.e., one
/// with erased regions), but this function does not.
pub fn trivial_dropck_outlives<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // None of these types have a destructor and hence they do not
        // require anything in particular to outlive the dtor's
        // execution.
        ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Bool
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Never
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Char
        | ty::GeneratorWitness(..)
        | ty::RawPtr(_)
        | ty::Ref(..)
        | ty::Str
        | ty::Foreign(..)
        | ty::Error(_) => true,

        // [T; N] and [T] have same properties as T.
        ty::Array(ty, _) | ty::Slice(ty) => trivial_dropck_outlives(tcx, ty),

        // (T1..Tn) and closures have same properties as T1..Tn --
        // check if *any* of those are trivial.
        ty::Tuple(ref tys) => tys.iter().all(|t| trivial_dropck_outlives(tcx, t.expect_ty())),
        ty::Closure(_, ref substs) => {
            substs.as_closure().upvar_tys().all(|t| trivial_dropck_outlives(tcx, t))
        }

        ty::Adt(def, _) => {
            if Some(def.did) == tcx.lang_items().manually_drop() {
                // `ManuallyDrop` never has a dtor.
                true
            } else {
                // Other types might. Moreover, PhantomData doesn't
                // have a dtor, but it is considered to own its
                // content, so it is non-trivial. Unions can have `impl Drop`,
                // and hence are non-trivial as well.
                false
            }
        }

        // The following *might* require a destructor: needs deeper inspection.
        ty::Dynamic(..)
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Opaque(..)
        | ty::Placeholder(..)
        | ty::Infer(_)
        | ty::Bound(..)
        | ty::Generator(..) => false,
    }
}

#[derive(Debug, HashStable)]
pub struct CandidateStep<'tcx> {
    pub self_ty: Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
    pub autoderefs: usize,
    /// `true` if the type results from a dereference of a raw pointer.
    /// when assembling candidates, we include these steps, but not when
    /// picking methods. This so that if we have `foo: *const Foo` and `Foo` has methods
    /// `fn by_raw_ptr(self: *const Self)` and `fn by_ref(&self)`, then
    /// `foo.by_raw_ptr()` will work and `foo.by_ref()` won't.
    pub from_unsafe_deref: bool,
    pub unsize: bool,
}

#[derive(Clone, Debug, HashStable)]
pub struct MethodAutoderefStepsResult<'tcx> {
    /// The valid autoderef steps that could be find.
    pub steps: Lrc<Vec<CandidateStep<'tcx>>>,
    /// If Some(T), a type autoderef reported an error on.
    pub opt_bad_ty: Option<Lrc<MethodAutoderefBadTy<'tcx>>>,
    /// If `true`, `steps` has been truncated due to reaching the
    /// recursion limit.
    pub reached_recursion_limit: bool,
}

#[derive(Debug, HashStable)]
pub struct MethodAutoderefBadTy<'tcx> {
    pub reached_raw_pointer: bool,
    pub ty: Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
}

/// Result from the `normalize_projection_ty` query.
#[derive(Clone, Debug, HashStable, TypeFoldable, Lift)]
pub struct NormalizationResult<'tcx> {
    /// Result of normalization.
    pub normalized_ty: Ty<'tcx>,
}

/// Outlives bounds are relationships between generic parameters,
/// whether they both be regions (`'a: 'b`) or whether types are
/// involved (`T: 'a`). These relationships can be extracted from the
/// full set of predicates we understand or also from types (in which
/// case they are called implied bounds). They are fed to the
/// `OutlivesEnv` which in turn is supplied to the region checker and
/// other parts of the inference system.
#[derive(Clone, Debug, TypeFoldable, Lift)]
pub enum OutlivesBound<'tcx> {
    RegionSubRegion(ty::Region<'tcx>, ty::Region<'tcx>),
    RegionSubParam(ty::Region<'tcx>, ty::ParamTy),
    RegionSubProjection(ty::Region<'tcx>, ty::ProjectionTy<'tcx>),
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for OutlivesBound<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            OutlivesBound::RegionSubRegion(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            OutlivesBound::RegionSubParam(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            OutlivesBound::RegionSubProjection(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
        }
    }
}
