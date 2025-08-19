use std::num::NonZero;

use rustc_macros::{Decodable, Encodable, HashStable_Generic, PrintAttribute};
use rustc_span::{ErrorGuaranteed, Symbol, sym};

use crate::RustcVersion;
use crate::attrs::PrintAttribute;

/// The version placeholder that recently stabilized features contain inside the
/// `since` field of the `#[stable]` attribute.
///
/// For more, see [this pull request](https://github.com/rust-lang/rust/pull/100591).
pub const VERSION_PLACEHOLDER: &str = concat!("CURRENT_RUSTC_VERSIO", "N");
// Note that the `concat!` macro above prevents `src/tools/replace-version-placeholder` from
// replacing the constant with the current version. Hardcoding the tool to skip this file doesn't
// work as the file can (and at some point will) be moved around.
//
// Turning the `concat!` macro into a string literal will make Pietro cry. That'd be sad :(

/// Represents the following attributes:
///
/// - `#[stable]`
/// - `#[unstable]`
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
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
#[derive(HashStable_Generic, PrintAttribute)]
pub struct ConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
    /// This is true iff the `const_stable_indirect` attribute is present.
    pub const_stable_indirect: bool,
}

impl ConstStability {
    pub fn from_partial(
        PartialConstStability { level, feature, promotable }: PartialConstStability,
        const_stable_indirect: bool,
    ) -> Self {
        Self { const_stable_indirect, level, feature, promotable }
    }

    /// The stability assigned to unmarked items when -Zforce-unstable-if-unmarked is set.
    pub fn unmarked(const_stable_indirect: bool, regular_stab: Stability) -> Self {
        Self {
            feature: regular_stab.feature,
            promotable: false,
            level: regular_stab.level,
            const_stable_indirect,
        }
    }

    pub fn is_const_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_const_stable(&self) -> bool {
        self.level.is_stable()
    }
}

/// Excludes `const_stable_indirect`. This is necessary because when `-Zforce-unstable-if-unmarked`
/// is set, we need to encode standalone `#[rustc_const_stable_indirect]` attributes
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
pub struct PartialConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
}

impl PartialConstStability {
    pub fn is_const_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_const_stable(&self) -> bool {
        self.level.is_stable()
    }
}

/// The available stability levels.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
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
        old_name: Option<Symbol>,
    },
    /// `#[stable]`
    Stable {
        /// Rust release which stabilized this feature.
        since: StableSince,
        /// This is `Some` if this item allowed to be referred to on stable via unstable modules;
        /// the `Symbol` is the deprecation message printed in that case.
        allowed_through_unstable_modules: Option<Symbol>,
    },
}

/// Rust release in which a feature is stabilized.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
pub enum StableSince {
    /// also stores the original symbol for printing
    Version(RustcVersion),
    /// Stabilized in the upcoming version, whatever number that is.
    Current,
    /// Failed to parse a stabilization version.
    Err(ErrorGuaranteed),
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
#[derive(HashStable_Generic, PrintAttribute)]
pub enum UnstableReason {
    None,
    Default,
    Some(Symbol),
}

/// Represents the `#[rustc_default_body_unstable]` attribute.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
pub struct DefaultBodyStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
}

impl UnstableReason {
    pub fn from_opt_reason(reason: Option<Symbol>) -> Self {
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
