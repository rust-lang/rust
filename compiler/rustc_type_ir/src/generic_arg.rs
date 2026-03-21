use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::GenericTypeVisitable;

use crate::Interner;

/// Raw data for an outlives relationship between two region positions.
/// `Outlives { longer, shorter }` means the region at position `longer`
/// outlives the region at position `shorter`. `usize::MAX` represents
/// `'static`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct OutlivesArgData {
    pub longer: usize,
    pub shorter: usize,
}

#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum GenericArgKind<I: Interner> {
    Lifetime(I::Region),
    Type(I::Ty),
    Const(I::Const),
    /// An outlives relation between two region positions within an
    /// instance's generic arg list. The interned `OutlivesArg` wraps an
    /// `OutlivesArgData` holding the two position indices.
    Outlives(I::OutlivesArg),
}

impl<I: Interner> Eq for GenericArgKind<I> {}

#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum TermKind<I: Interner> {
    Ty(I::Ty),
    Const(I::Const),
}

impl<I: Interner> Eq for TermKind<I> {}
