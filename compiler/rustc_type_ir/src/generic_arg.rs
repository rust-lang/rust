use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::GenericTypeVisitable;

use crate::{Interner, Ty};

#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum GenericArgKind<I: Interner> {
    Lifetime(I::Region),
    Type(Ty<I>),
    Const(I::Const),
}

impl<I: Interner> Eq for GenericArgKind<I> {}

#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum TermKind<I: Interner> {
    Ty(Ty<I>),
    Const(I::Const),
}

impl<I: Interner> Eq for TermKind<I> {}
