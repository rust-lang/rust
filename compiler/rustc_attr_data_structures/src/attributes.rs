use rustc_abi::Align;
use rustc_ast::token::CommentKind;
use rustc_ast::{self as ast, AttrStyle};
use rustc_macros::{Decodable, Encodable, HashStable_Generic, PrintAttribute};
use rustc_span::hygiene::Transparency;
use rustc_span::{Span, Symbol};
use thin_vec::ThinVec;

use crate::{DefaultBodyStability, PartialConstStability, PrintAttribute, RustcVersion, Stability};

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
    /// `#[rustc_force_inline]` forces inlining to happen in the MIR inliner - it reports an error
    /// if the inlining cannot happen. It is limited to only free functions so that the calls
    /// can always be resolved.
    Force {
        attr_span: Span,
        reason: Option<Symbol>,
    },
}

impl InlineAttr {
    pub fn always(&self) -> bool {
        match self {
            InlineAttr::Always | InlineAttr::Force { .. } => true,
            InlineAttr::None | InlineAttr::Hint | InlineAttr::Never => false,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq, HashStable_Generic)]
pub enum InstructionSetAttr {
    ArmA32,
    ArmT32,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq, HashStable_Generic, Default)]
pub enum OptimizeAttr {
    /// No `#[optimize(..)]` attribute
    #[default]
    Default,
    /// `#[optimize(none)]`
    DoNotOptimize,
    /// `#[optimize(speed)]`
    Speed,
    /// `#[optimize(size)]`
    Size,
}

impl OptimizeAttr {
    pub fn do_not_optimize(&self) -> bool {
        matches!(self, Self::DoNotOptimize)
    }
}

#[derive(PartialEq, Debug, Encodable, Decodable, Copy, Clone, HashStable_Generic, PrintAttribute)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprRust,
    ReprC,
    ReprPacked(Align),
    ReprSimd,
    ReprTransparent,
    ReprAlign(Align),
    // this one is just so we can emit a lint for it
    ReprEmpty,
}
pub use ReprAttr::*;

pub enum TransparencyError {
    UnknownTransparency(Symbol, Span),
    MultipleTransparencyAttrs(Span, Span),
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
#[derive(Encodable, Decodable, HashStable_Generic, PrintAttribute)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy),
}

#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic, PrintAttribute)]
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
#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic, PrintAttribute)]
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

/// Represent parsed, *built in*, inert attributes.
///
/// That means attributes that are not actually ever expanded.
/// For more information on this, see the module docs on the [`rustc_attr_parsing`] crate.
/// They're instead used as markers, to guide the compilation process in various way in most every stage of the compiler.
/// These are kept around after the AST, into the HIR and further on.
///
/// The word "parsed" could be a little misleading here, because the parser already parses
/// attributes early on. However, the result, an [`ast::Attribute`]
/// is only parsed at a high level, still containing a token stream in many cases. That is
/// because the structure of the contents varies from attribute to attribute.
/// With a parsed attribute I mean that each attribute is processed individually into a
/// final structure, which on-site (the place where the attribute is useful for, think the
/// the place where `must_use` is checked) little to no extra parsing or validating needs to
/// happen.
///
/// For more docs, look in [`rustc_attr_parsing`].
///
/// [`rustc_attr_parsing`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/index.html
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum AttributeKind {
    // tidy-alphabetical-start
    /// Represents `#[rustc_allow_const_fn_unstable]`.
    AllowConstFnUnstable(ThinVec<Symbol>),

    /// Represents `#[allow_internal_unstable]`.
    AllowInternalUnstable(ThinVec<(Symbol, Span)>),

    /// Represents `#[rustc_default_body_unstable]`.
    BodyStability {
        stability: DefaultBodyStability,
        /// Span of the `#[rustc_default_body_unstable(...)]` attribute
        span: Span,
    },

    /// Represents `#[rustc_confusables]`.
    Confusables {
        symbols: ThinVec<Symbol>,
        // FIXME(jdonszelmann): remove when target validation code is moved
        first_span: Span,
    },

    /// Represents `#[rustc_const_stable]` and `#[rustc_const_unstable]`.
    ConstStability {
        stability: PartialConstStability,
        /// Span of the `#[rustc_const_stable(...)]` or `#[rustc_const_unstable(...)]` attribute
        span: Span,
    },

    /// Represents `#[rustc_const_stable_indirect]`.
    ConstStabilityIndirect,

    /// Represents [`#[deprecated]`](https://doc.rust-lang.org/stable/reference/attributes/diagnostics.html#the-deprecated-attribute).
    Deprecation { deprecation: Deprecation, span: Span },

    /// Represents [`#[doc]`](https://doc.rust-lang.org/stable/rustdoc/write-documentation/the-doc-attribute.html).
    DocComment { style: AttrStyle, kind: CommentKind, span: Span, comment: Symbol },

    /// Represents `#[rustc_macro_transparency]`.
    MacroTransparency(Transparency),

    /// Represents [`#[repr]`](https://doc.rust-lang.org/stable/reference/type-layout.html#representations).
    Repr(ThinVec<(ReprAttr, Span)>),

    /// Represents `#[stable]`, `#[unstable]` and `#[rustc_allowed_through_unstable_modules]`.
    Stability {
        stability: Stability,
        /// Span of the attribute.
        span: Span,
    },
    // tidy-alphabetical-end
}
