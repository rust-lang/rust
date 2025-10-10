use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::Interner;

#[derive_where(Clone, Copy, Hash, PartialEq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum PatternKind<I: Interner> {
    Range { start: I::Const, end: I::Const },
    Or(I::PatList),
}

impl<I: Interner> Eq for PatternKind<I> {}
