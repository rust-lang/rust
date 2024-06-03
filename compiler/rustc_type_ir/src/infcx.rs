use crate::{ConstVid, EffectVid, FloatVid, IntVid, Interner, RegionVid, TyVid, UniverseIndex};

pub trait InferCtxtLike {
    type Interner: Interner;

    fn interner(&self) -> Self::Interner;

    fn universe_of_ty(&self, ty: TyVid) -> Option<UniverseIndex>;
    fn universe_of_lt(&self, lt: RegionVid) -> Option<UniverseIndex>;
    fn universe_of_ct(&self, ct: ConstVid) -> Option<UniverseIndex>;

    fn opportunistic_resolve_ty_var(&self, vid: TyVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_int_var(&self, vid: IntVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_float_var(&self, vid: FloatVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_ct_var(&self, vid: ConstVid) -> <Self::Interner as Interner>::Const;
    fn opportunistic_resolve_effect_var(
        &self,
        vid: EffectVid,
    ) -> <Self::Interner as Interner>::Const;
    fn opportunistic_resolve_lt_var(&self, vid: RegionVid) -> <Self::Interner as Interner>::Region;

    fn defining_opaque_types(&self) -> <Self::Interner as Interner>::DefiningOpaqueTypes;
}
