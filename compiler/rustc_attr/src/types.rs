use std::num::NonZero;

use rustc_abi::Align;
use rustc_ast as ast;
use rustc_macros::{Encodable, Decodable, HashStable_Generic};
use rustc_session::RustcVersion;
use rustc_span::{sym, Span, Symbol};

/// The version placeholder that recently stabilized features contain inside the
/// `since` field of the `#[stable]` attribute.
///
/// For more, see [this pull request](https://github.com/rust-lang/rust/pull/100591).
pub const VERSION_PLACEHOLDER: &str = "CURRENT_RUSTC_VERSION";

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

/// Represents the following attributes:
///
/// - `#[stable]`
/// - `#[unstable]`
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
}

impl Stability {
    pub fn is_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_stable(&self) -> bool {
        self.level.is_stable()
    }

    pub fn stable_since(&self) -> Option<StableSince> {
        self.level.stable_since()
    }
}

/// Represents the `#[rustc_const_unstable]` and `#[rustc_const_stable]` attributes.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct ConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// This is true iff the `const_stable_indirect` attribute is present.
    pub const_stable_indirect: bool,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
}

impl ConstStability {
    pub fn is_const_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_const_stable(&self) -> bool {
        self.level.is_stable()
    }
}

/// Represents the `#[rustc_default_body_unstable]` attribute.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct DefaultBodyStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
}

/// The available stability levels.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, Hash)]
#[derive(HashStable_Generic)]
pub enum StabilityLevel {
    /// `#[unstable]`
    Unstable {
        /// Reason for the current stability level.
        reason: UnstableReason,
        /// Relevant `rust-lang/rust` issue.
        issue: Option<NonZero<u32>>,
        is_soft: bool,
        /// If part of a feature is stabilized and a new feature is added for the remaining parts,
        /// then the `implied_by` attribute is used to indicate which now-stable feature previously
        /// contained an item.
        ///
        /// ```pseudo-Rust
        /// #[unstable(feature = "foo", issue = "...")]
        /// fn foo() {}
        /// #[unstable(feature = "foo", issue = "...")]
        /// fn foobar() {}
        /// ```
        ///
        /// ...becomes...
        ///
        /// ```pseudo-Rust
        /// #[stable(feature = "foo", since = "1.XX.X")]
        /// fn foo() {}
        /// #[unstable(feature = "foobar", issue = "...", implied_by = "foo")]
        /// fn foobar() {}
        /// ```
        implied_by: Option<Symbol>,
    },
    /// `#[stable]`
    Stable {
        /// Rust release which stabilized this feature.
        since: StableSince,
        /// Is this item allowed to be referred to on stable, despite being contained in unstable
        /// modules?
        allowed_through_unstable_modules: bool,
    },
}

/// Rust release in which a feature is stabilized.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable_Generic)]
pub enum StableSince {
    Version(RustcVersion),
    /// Stabilized in the upcoming version, whatever number that is.
    Current,
    /// Failed to parse a stabilization version.
    Err,
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool {
        matches!(self, StabilityLevel::Unstable { .. })
    }
    pub fn is_stable(&self) -> bool {
        matches!(self, StabilityLevel::Stable { .. })
    }
    pub fn stable_since(&self) -> Option<StableSince> {
        match *self {
            StabilityLevel::Stable { since, .. } => Some(since),
            StabilityLevel::Unstable { .. } => None,
        }
    }
}

#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, Hash)]
#[derive(HashStable_Generic)]
pub enum UnstableReason {
    None,
    Default,
    Some(Symbol),
}

impl UnstableReason {
    pub(crate) fn from_opt_reason(reason: Option<Symbol>) -> Self {
        // UnstableReason::Default constructed manually
        match reason {
            Some(r) => Self::Some(r),
            None => Self::None,
        }
    }

    pub fn to_opt_reason(&self) -> Option<Symbol> {
        match self {
            Self::None => None,
            Self::Default => Some(sym::unstable_location_reason_default),
            Self::Some(r) => Some(*r),
        }
    }
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
