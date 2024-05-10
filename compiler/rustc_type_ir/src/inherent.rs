use crate::{BoundVar, DebruijnIndex, Interner, UniverseIndex};

pub trait Ty<I: Interner<Ty = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait Region<I: Interner<Region = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;

    fn new_static(interner: I) -> Self;
}

pub trait Const<I: Interner<Const = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar, ty: I::Ty) -> Self;

    fn ty(self) -> I::Ty;
}

pub trait GenericsOf<I: Interner> {
    fn count(&self) -> usize;
}

pub trait GenericArgs<I: Interner> {
    fn type_at(self, i: usize) -> I::Ty;

    fn identity_for_item(interner: I, def_id: I::DefId) -> I::GenericArgs;
}

/// Common capabilities of placeholder kinds
pub trait PlaceholderLike {
    fn universe(self) -> UniverseIndex;
    fn var(self) -> BoundVar;

    fn with_updated_universe(self, ui: UniverseIndex) -> Self;

    fn new(ui: UniverseIndex, var: BoundVar) -> Self;
}

pub trait IntoKind {
    type Kind;

    fn kind(self) -> Self::Kind;
}

pub trait BoundVars<I: Interner> {
    fn bound_vars(&self) -> I::BoundVars;

    fn has_no_bound_vars(&self) -> bool;
}
