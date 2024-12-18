use rustc_abi::Align;
use rustc_ast as ast;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::{Span, Symbol};

use crate::RustcVersion;

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq, HashStable_Generic)]
pub enum InstructionSetAttr {
    ArmA32,
    ArmT32,
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum OptimizeAttr {
    None,
    Speed,
    Size,
}

#[derive(Clone, Debug, Encodable, Decodable)]
pub enum DiagnosticAttribute {
    // tidy-alphabetical-start
    DoNotRecommend,
    OnUnimplemented,
    // tidy-alphabetical-end
}

#[derive(PartialEq, Debug, Encodable, Decodable, Copy, Clone)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprRust,
    ReprC,
    ReprPacked(Align),
    ReprSimd,
    ReprTransparent,
    ReprAlign(Align),
}
pub use ReprAttr::*;

pub enum TransparencyError {
    UnknownTransparency(Symbol, Span),
    MultipleTransparencyAttrs(Span, Span),
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
#[derive(Encodable, Decodable)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy),
}

#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic)]
pub struct Deprecation {
    pub since: DeprecatedSince,
    /// The note to issue a reason.
    pub note: Option<Symbol>,
    /// A text snippet used to completely replace any use of the deprecated item in an expression.
    ///
    /// This is currently unstable.
    pub suggestion: Option<Symbol>,
}

/// Release in which an API is deprecated.
#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic)]
pub enum DeprecatedSince {
    RustcVersion(RustcVersion),
    /// Deprecated in the future ("to be determined").
    Future,
    /// `feature(staged_api)` is off. Deprecation versions outside the standard
    /// library are allowed to be arbitrary strings, for better or worse.
    NonStandard(Symbol),
    /// Deprecation version is unspecified but optional.
    Unspecified,
    /// Failed to parse a deprecation version, or the deprecation version is
    /// unspecified and required. An error has already been emitted.
    Err,
}

impl Deprecation {
    /// Whether an item marked with #[deprecated(since = "X")] is currently
    /// deprecated (i.e., whether X is not greater than the current rustc
    /// version).
    pub fn is_in_effect(&self) -> bool {
        match self.since {
            DeprecatedSince::RustcVersion(since) => since <= RustcVersion::CURRENT,
            DeprecatedSince::Future => false,
            // The `since` field doesn't have semantic purpose without `#![staged_api]`.
            DeprecatedSince::NonStandard(_) => true,
            // Assume deprecation is in effect if "since" field is absent or invalid.
            DeprecatedSince::Unspecified | DeprecatedSince::Err => true,
        }
    }

    pub fn is_since_rustc_version(&self) -> bool {
        matches!(self.since, DeprecatedSince::RustcVersion(_))
    }
}
