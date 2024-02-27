use crate::{BoundVar, DebruijnIndex, Interner};

pub trait Ty<I: Interner<Ty = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait Region<I: Interner<Region = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait Const<I: Interner<Const = Self>> {
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar, ty: I::Ty) -> Self;
}
