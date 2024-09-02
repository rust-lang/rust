//! Machinery for hygienic macros.
//!
//! Inspired by Matthew Flatt et al., “Macros That Work Together: Compile-Time Bindings, Partial
//! Expansion, and Definition Contexts,” *Journal of Functional Programming* 22, no. 2
//! (March 1, 2012): 181–216, <https://doi.org/10.1017/S0956796812000093>.
//!
//! Also see <https://rustc-dev-guide.rust-lang.org/macro-expansion.html#hygiene-and-hierarchies>
//!
//! # The Expansion Order Hierarchy
//!
//! `ExpnData` in rustc, rust-analyzer's version is [`MacroCallLoc`]. Traversing the hierarchy
//! upwards can be achieved by walking up [`MacroCallLoc::kind`]'s contained file id, as
//! [`MacroFile`]s are interned [`MacroCallLoc`]s.
//!
//! # The Macro Definition Hierarchy
//!
//! `SyntaxContextData` in rustc and rust-analyzer. Basically the same in both.
//!
//! # The Call-site Hierarchy
//!
//! `ExpnData::call_site` in rustc, [`MacroCallLoc::call_site`] in rust-analyzer.
use std::fmt;

use salsa::{InternId, InternValue};

use crate::MacroCallId;

/// Interned [`SyntaxContextData`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxContextId(InternId);

impl fmt::Debug for SyntaxContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "{}", self.0.as_u32())
        } else {
            f.debug_tuple("SyntaxContextId").field(&self.0).finish()
        }
    }
}

impl salsa::InternKey for SyntaxContextId {
    fn from_intern_id(v: salsa::InternId) -> Self {
        SyntaxContextId(v)
    }
    fn as_intern_id(&self) -> salsa::InternId {
        self.0
    }
}

impl fmt::Display for SyntaxContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.as_u32())
    }
}

impl SyntaxContextId {
    /// The root context, which is the parent of all other contexts. All [`FileId`]s have this context.
    pub const ROOT: Self = SyntaxContextId(unsafe { InternId::new_unchecked(0) });

    pub fn is_root(self) -> bool {
        self == Self::ROOT
    }

    /// Deconstruct a `SyntaxContextId` into a raw `u32`.
    /// This should only be used for deserialization purposes for the proc-macro server.
    pub fn into_u32(self) -> u32 {
        self.0.as_u32()
    }

    /// Constructs a `SyntaxContextId` from a raw `u32`.
    /// This should only be used for serialization purposes for the proc-macro server.
    pub fn from_u32(u32: u32) -> Self {
        Self(InternId::from(u32))
    }
}

/// A syntax context describes a hierarchy tracking order of macro definitions.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct SyntaxContextData {
    /// Invariant: Only [`SyntaxContextId::ROOT`] has a [`None`] outer expansion.
    // FIXME: The None case needs to encode the context crate id. We can encode that as the MSB of
    // MacroCallId is reserved anyways so we can do bit tagging here just fine.
    // The bigger issue is that this will cause interning to now create completely separate chains
    // per crate. Though that is likely not a problem as `MacroCallId`s are already crate calling dependent.
    pub outer_expn: Option<MacroCallId>,
    pub outer_transparency: Transparency,
    pub parent: SyntaxContextId,
    /// This context, but with all transparent and semi-transparent expansions filtered away.
    pub opaque: SyntaxContextId,
    /// This context, but with all transparent expansions filtered away.
    pub opaque_and_semitransparent: SyntaxContextId,
}

impl InternValue for SyntaxContextData {
    type Key = (SyntaxContextId, Option<MacroCallId>, Transparency);

    fn into_key(&self) -> Self::Key {
        (self.parent, self.outer_expn, self.outer_transparency)
    }
}

impl std::fmt::Debug for SyntaxContextData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntaxContextData")
            .field("outer_expn", &self.outer_expn)
            .field("outer_transparency", &self.outer_transparency)
            .field("parent", &self.parent)
            .field("opaque", &self.opaque)
            .field("opaque_and_semitransparent", &self.opaque_and_semitransparent)
            .finish()
    }
}

impl SyntaxContextData {
    pub fn root() -> Self {
        SyntaxContextData {
            outer_expn: None,
            outer_transparency: Transparency::Opaque,
            parent: SyntaxContextId::ROOT,
            opaque: SyntaxContextId::ROOT,
            opaque_and_semitransparent: SyntaxContextId::ROOT,
        }
    }
}

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    Transparent,
    /// Identifier produced by a semi-transparent expansion may be resolved
    /// either at call-site or at definition-site.
    /// If it's a local variable, label or `$crate` then it's resolved at def-site.
    /// Otherwise it's resolved at call-site.
    /// `macro_rules` macros behave like this, built-in macros currently behave like this too,
    /// but that's an implementation detail.
    SemiTransparent,
    /// Identifier produced by an opaque expansion is always resolved at definition-site.
    /// Def-site spans in procedural macros, identifiers from `macro` by default use this.
    Opaque,
}
