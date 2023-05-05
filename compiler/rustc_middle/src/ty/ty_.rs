use crate::ty::{BoundTy, Interned, Placeholder, TyKind, WithCachedTypeInfo};

/// Use this rather than `TyKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
pub struct Ty<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>);

pub type PlaceholderType = Placeholder<BoundTy>;

#[derive(Debug, Default, Copy, Clone)]
pub struct InferVarInfo {
    /// This is true if we identified that this Ty (`?T`) is found in a `?T: Foo`
    /// obligation, where:
    ///
    ///  * `Foo` is not `Sized`
    ///  * `(): Foo` may be satisfied
    pub self_in_trait: bool,
    /// This is true if we identified that this Ty (`?T`) is found in a `<_ as
    /// _>::AssocType = ?T`
    pub output: bool,
}

// FIXME: move `Ty` inherent impls here (and possibly change structure wrt `sty` mod)
