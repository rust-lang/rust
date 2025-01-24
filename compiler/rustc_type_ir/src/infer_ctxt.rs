use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};

use crate::fold::TypeFoldable;
use crate::relate::RelateResult;
use crate::relate::combine::PredicateEmittingRelation;
use crate::{self as ty, Interner};

/// The current typing mode of an inference context. We unfortunately have some
/// slightly different typing rules depending on the current context. See the
/// doc comment for each variant for how and why they are used.
///
/// In most cases you can get the correct typing mode automically via:
/// - `mir::Body::typing_mode`
/// - `rustc_lint::LateContext::typing_mode`
///
/// If neither of these functions are available, feel free to reach out to
/// t-types for help.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub enum TypingMode<I: Interner> {
    /// When checking whether impls overlap, we check whether any obligations
    /// are guaranteed to never hold when unifying the impls. This requires us
    /// to be complete: we must never fail to prove something which may actually
    /// hold.
    ///
    /// In this typing mode we bail with ambiguity in case its not knowable
    /// whether a trait goal may hold, e.g. because the trait may get implemented
    /// in a downstream or sibling crate.
    ///
    /// We also have to be careful when generalizing aliases inside of higher-ranked
    /// types to not unnecessarily constrain any inference variables.
    Coherence,
    /// Analysis includes type inference, checking that items are well-formed, and
    /// pretty much everything else which may emit proper type errors to the user.
    ///
    /// We only normalize opaque types which may get defined by the current body,
    /// which are stored in `defining_opaque_types`.
    ///
    /// We also refuse to project any associated type that is marked `default`.
    /// Non-`default` ("final") types are always projected. This is necessary in
    /// general for soundness of specialization. However, we *could* allow projections
    /// in fully-monomorphic cases. We choose not to, because we prefer for `default type`
    /// to force the type definition to be treated abstractly by any consumers of the
    /// impl. Concretely, that means that the following example will
    /// fail to compile:
    ///
    /// ```compile_fail,E0308
    /// #![feature(specialization)]
    /// trait Assoc {
    ///     type Output;
    /// }
    ///
    /// impl<T> Assoc for T {
    ///     default type Output = bool;
    /// }
    ///
    /// fn main() {
    ///     let x: <() as Assoc>::Output = true;
    /// }
    /// ```
    Analysis { defining_opaque_types: I::DefiningOpaqueTypes },
    /// Any analysis after borrowck for a given body should be able to use all the
    /// hidden types defined by borrowck, without being able to define any new ones.
    ///
    /// This is currently only used by the new solver, but should be implemented in
    /// the old solver as well.
    PostBorrowckAnalysis { defined_opaque_types: I::DefiningOpaqueTypes },
    /// After analysis, mostly during codegen and MIR optimizations, we're able to
    /// reveal all opaque types. As the concrete type should *never* be observable
    /// directly by the user, this should not be used by checks which may expose
    /// such details to the user.
    ///
    /// There are some exceptions to this as for example `layout_of` and const-evaluation
    /// always run in `PostAnalysis` mode, even when used during analysis. This exposes
    /// some information about the underlying type to users, but not the type itself.
    PostAnalysis,
}

impl<I: Interner> TypingMode<I> {
    /// Analysis outside of a body does not define any opaque types.
    pub fn non_body_analysis() -> TypingMode<I> {
        TypingMode::Analysis { defining_opaque_types: Default::default() }
    }

    /// While typechecking a body, we need to be able to define the opaque
    /// types defined by that body.
    pub fn analysis_in_body(cx: I, body_def_id: I::LocalDefId) -> TypingMode<I> {
        TypingMode::Analysis { defining_opaque_types: cx.opaque_types_defined_by(body_def_id) }
    }

    pub fn post_borrowck_analysis(cx: I, body_def_id: I::LocalDefId) -> TypingMode<I> {
        TypingMode::PostBorrowckAnalysis {
            defined_opaque_types: cx.opaque_types_defined_by(body_def_id),
        }
    }
}

pub trait InferCtxtLike: Sized {
    type Interner: Interner;
    fn cx(&self) -> Self::Interner;

    /// Whether the new trait solver is enabled. This only exists because rustc
    /// shares code between the new and old trait solvers; for all other users,
    /// this should always be true. If this is unknowingly false and you try to
    /// use the new trait solver, things will break badly.
    fn next_trait_solver(&self) -> bool {
        true
    }

    fn typing_mode(&self) -> TypingMode<Self::Interner>;

    fn universe(&self) -> ty::UniverseIndex;
    fn create_next_universe(&self) -> ty::UniverseIndex;

    fn universe_of_ty(&self, ty: ty::TyVid) -> Option<ty::UniverseIndex>;
    fn universe_of_lt(&self, lt: ty::RegionVid) -> Option<ty::UniverseIndex>;
    fn universe_of_ct(&self, ct: ty::ConstVid) -> Option<ty::UniverseIndex>;

    fn root_ty_var(&self, var: ty::TyVid) -> ty::TyVid;
    fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid;

    fn opportunistic_resolve_ty_var(&self, vid: ty::TyVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_int_var(&self, vid: ty::IntVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_float_var(
        &self,
        vid: ty::FloatVid,
    ) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_ct_var(
        &self,
        vid: ty::ConstVid,
    ) -> <Self::Interner as Interner>::Const;
    fn opportunistic_resolve_lt_var(
        &self,
        vid: ty::RegionVid,
    ) -> <Self::Interner as Interner>::Region;

    fn next_region_infer(&self) -> <Self::Interner as Interner>::Region;
    fn next_ty_infer(&self) -> <Self::Interner as Interner>::Ty;
    fn next_const_infer(&self) -> <Self::Interner as Interner>::Const;
    fn fresh_args_for_item(
        &self,
        def_id: <Self::Interner as Interner>::DefId,
    ) -> <Self::Interner as Interner>::GenericArgs;

    fn instantiate_binder_with_infer<T: TypeFoldable<Self::Interner> + Copy>(
        &self,
        value: ty::Binder<Self::Interner, T>,
    ) -> T;

    fn enter_forall<T: TypeFoldable<Self::Interner> + Copy, U>(
        &self,
        value: ty::Binder<Self::Interner, T>,
        f: impl FnOnce(T) -> U,
    ) -> U;

    fn equate_ty_vids_raw(&self, a: ty::TyVid, b: ty::TyVid);
    fn equate_int_vids_raw(&self, a: ty::IntVid, b: ty::IntVid);
    fn equate_float_vids_raw(&self, a: ty::FloatVid, b: ty::FloatVid);
    fn equate_const_vids_raw(&self, a: ty::ConstVid, b: ty::ConstVid);

    fn instantiate_ty_var_raw<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: ty::TyVid,
        instantiation_variance: ty::Variance,
        source_ty: <Self::Interner as Interner>::Ty,
    ) -> RelateResult<Self::Interner, ()>;
    fn instantiate_int_var_raw(&self, vid: ty::IntVid, value: ty::IntVarValue);
    fn instantiate_float_var_raw(&self, vid: ty::FloatVid, value: ty::FloatVarValue);
    fn instantiate_const_var_raw<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: ty::ConstVid,
        source_ct: <Self::Interner as Interner>::Const,
    ) -> RelateResult<Self::Interner, ()>;

    fn set_tainted_by_errors(&self, e: <Self::Interner as Interner>::ErrorGuaranteed);

    fn shallow_resolve(
        &self,
        ty: <Self::Interner as Interner>::Ty,
    ) -> <Self::Interner as Interner>::Ty;
    fn shallow_resolve_const(
        &self,
        ty: <Self::Interner as Interner>::Const,
    ) -> <Self::Interner as Interner>::Const;

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<Self::Interner>;

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T;

    fn sub_regions(
        &self,
        sub: <Self::Interner as Interner>::Region,
        sup: <Self::Interner as Interner>::Region,
    );

    fn equate_regions(
        &self,
        a: <Self::Interner as Interner>::Region,
        b: <Self::Interner as Interner>::Region,
    );

    fn register_ty_outlives(
        &self,
        ty: <Self::Interner as Interner>::Ty,
        r: <Self::Interner as Interner>::Region,
    );
}
