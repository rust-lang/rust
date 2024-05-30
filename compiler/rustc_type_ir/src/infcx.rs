use crate::fold::TypeFoldable;
use crate::{self as ty, Interner};

pub trait InferCtxtLike {
    type Interner: Interner;

    fn interner(&self) -> Self::Interner;

    fn universe_of_ty(&self, ty: ty::TyVid) -> Option<ty::UniverseIndex>;
    fn universe_of_lt(&self, lt: ty::RegionVid) -> Option<ty::UniverseIndex>;
    fn universe_of_ct(&self, ct: ty::ConstVid) -> Option<ty::UniverseIndex>;

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

    fn instantiate_binder_with_infer<T: TypeFoldable<Self::Interner> + Copy>(
        &self,
        value: ty::Binder<Self::Interner, T>,
    ) -> T;

    fn enter_forall<T: TypeFoldable<Self::Interner> + Copy, U>(
        &self,
        value: ty::Binder<Self::Interner, T>,
        f: impl FnOnce(T) -> U,
    ) -> U;
}
