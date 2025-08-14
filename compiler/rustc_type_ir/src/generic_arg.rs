use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};

use crate::Interner;

#[derive_where(Clone, Copy, PartialEq, Eq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum GenericArgKind<I: Interner> {
    Lifetime(I::Region),
    Type(I::Ty),
    Const(I::Const),
}

#[derive_where(Clone, Copy, PartialEq, Eq, Debug; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum TermKind<I: Interner> {
    Ty(I::Ty),
    Const(I::Const),
}
