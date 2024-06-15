use crate::fold::TypeFoldable;
use crate::relate::Relate;
use crate::solve::{Goal, NoSolution};
use crate::{self as ty, Interner};

pub trait InferCtxtLike: Sized {
    type Interner: Interner;

    fn interner(&self) -> Self::Interner;

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
    fn opportunistic_resolve_effect_var(
        &self,
        vid: ty::EffectVid,
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

    fn relate<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>;

    fn eq_structurally_relating_aliases<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>;

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<Self::Interner>;

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T;
}
