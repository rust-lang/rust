use crate::fold::TypeFoldable;
use crate::relate::RelateResult;
use crate::relate::combine::PredicateEmittingRelation;
use crate::solve::SolverMode;
use crate::{self as ty, Interner};

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

    fn solver_mode(&self) -> SolverMode;

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

    fn defining_opaque_types(&self) -> <Self::Interner as Interner>::DefiningOpaqueTypes;

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
