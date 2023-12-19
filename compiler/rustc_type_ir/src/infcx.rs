use crate::{ConstVid, Interner, TyVid, UniverseIndex};

pub trait InferCtxtLike {
    type Interner: Interner;

    fn interner(&self) -> Self::Interner;

    fn universe_of_ty(&self, ty: TyVid) -> Option<UniverseIndex>;

    /// Resolve `TyVid` to its root `TyVid`.
    fn root_ty_var(&self, vid: TyVid) -> TyVid;

    /// Resolve `TyVid` to its inferred type, if it has been equated with a non-infer type.
    fn probe_ty_var(&self, vid: TyVid) -> Option<<Self::Interner as Interner>::Ty>;

    fn universe_of_lt(
        &self,
        lt: <Self::Interner as Interner>::InferRegion,
    ) -> Option<UniverseIndex>;

    /// Resolve `InferRegion` to its inferred region, if it has been equated with
    /// a non-infer region.
    ///
    /// FIXME: This has slightly different semantics than `{probe,resolve}_{ty,ct}_var`,
    /// that has to do with the fact unlike `Ty` or `Const` vars, in rustc, we may
    /// not always be able to *name* the root region var from the universe of the
    /// var we're trying to resolve. That's why it's called *opportunistic*.
    fn opportunistic_resolve_lt_var(
        &self,
        vid: <Self::Interner as Interner>::InferRegion,
    ) -> Option<<Self::Interner as Interner>::Region>;

    fn universe_of_ct(&self, ct: ConstVid) -> Option<UniverseIndex>;

    /// Resolve `ConstVid` to its root `ConstVid`.
    fn root_ct_var(&self, vid: ConstVid) -> ConstVid;

    /// Resolve `ConstVid` to its inferred type, if it has been equated with a non-infer type.
    fn probe_ct_var(&self, vid: ConstVid) -> Option<<Self::Interner as Interner>::Const>;
}
