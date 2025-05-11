//! Common utilities shared by both `rustc_ast` and `rustc_type_ir`.
//!
//! Don't depend on this crate directly; both of those crates should re-export
//! the functionality. Additionally, if you're in scope of `rustc_middle`, then
//! prefer imports via that too, to avoid needing to directly depend on (e.g.)
//! `rustc_type_ir` for a single import.

// tidy-alphabetical-start
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(feature = "nightly", feature(never_type))]
#![cfg_attr(feature = "nightly", feature(rustc_attrs))]
// tidy-alphabetical-end

#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};

pub mod visit;

/// The movability of a coroutine / closure literal:
/// whether a coroutine contains self-references, causing it to be `!Unpin`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum Movability {
    /// May contain self-references, `!Unpin`.
    Static,
    /// Must not contain self-references, `Unpin`.
    Movable,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum Mutability {
    // N.B. Order is deliberate, so that Not < Mut
    Not,
    Mut,
}

impl Mutability {
    pub fn invert(self) -> Self {
        match self {
            Mutability::Mut => Mutability::Not,
            Mutability::Not => Mutability::Mut,
        }
    }

    /// Returns `""` (empty string) or `"mut "` depending on the mutability.
    pub fn prefix_str(self) -> &'static str {
        match self {
            Mutability::Mut => "mut ",
            Mutability::Not => "",
        }
    }

    /// Returns `"&"` or `"&mut "` depending on the mutability.
    pub fn ref_prefix_str(self) -> &'static str {
        match self {
            Mutability::Not => "&",
            Mutability::Mut => "&mut ",
        }
    }

    /// Returns `"const"` or `"mut"` depending on the mutability.
    pub fn ptr_str(self) -> &'static str {
        match self {
            Mutability::Not => "const",
            Mutability::Mut => "mut",
        }
    }

    /// Returns `""` (empty string) or `"mutably "` depending on the mutability.
    pub fn mutably_str(self) -> &'static str {
        match self {
            Mutability::Not => "",
            Mutability::Mut => "mutably ",
        }
    }

    /// Return `true` if self is mutable
    pub fn is_mut(self) -> bool {
        matches!(self, Self::Mut)
    }

    /// Return `true` if self is **not** mutable
    pub fn is_not(self) -> bool {
        matches!(self, Self::Not)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum Pinnedness {
    Not,
    Pinned,
}
