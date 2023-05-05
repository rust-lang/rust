use crate::ty::{BoundTy, Interned, Placeholder, TyKind, WithCachedTypeInfo};

/// Use this rather than `TyKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
pub struct Ty<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>);

pub type PlaceholderType = Placeholder<BoundTy>;

// FIXME: move `Ty` inherent impls here (and possibly change structure wrt `sty` mod)
