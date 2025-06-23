use rustc_abi::Align;
use rustc_ast::token::CommentKind;
use rustc_ast::{self as ast, AttrStyle};
use rustc_macros::{Decodable, Encodable, HashStable_Generic, PrintAttribute};
use rustc_span::hygiene::Transparency;
use rustc_span::{Span, Symbol};
use thin_vec::ThinVec;

use crate::{DefaultBodyStability, PartialConstStability, PrintAttribute, RustcVersion, Stability};

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic, PrintAttribute)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, PrintAttribute)]
#[derive(Encodable, Decodable, HashStable_Generic)]
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

/// Represents parsed *built-in* inert attributes.
///
/// ## Overview
/// These attributes are markers that guide the compilation process and are never expanded into other code.
/// They persist throughout the compilation phases, from AST to HIR and beyond.
///
/// ## Attribute Processing
/// While attributes are initially parsed by [`rustc_parse`] into [`ast::Attribute`], they still contain raw token streams
/// because different attributes have different internal structures. This enum represents the final,
/// fully parsed form of these attributes, where each variant contains contains all the information and
/// structure relevant for the specific attribute.
///
/// Some attributes can be applied multiple times to the same item, and they are "collapsed" into a single
/// semantic attribute. For example:
/// ```rust
/// #[repr(C)]
/// #[repr(packed)]
/// struct S { }
/// ```
/// This is equivalent to `#[repr(C, packed)]` and results in a single [`AttributeKind::Repr`] containing
/// both `C` and `packed` annotations. This collapsing happens during parsing and is reflected in the
/// data structures defined in this enum.
///
/// ## Usage
/// These parsed attributes are used throughout the compiler to:
/// - Control code generation (e.g., `#[repr]`)
/// - Mark API stability (`#[stable]`, `#[unstable]`)
/// - Provide documentation (`#[doc]`)
/// - Guide compiler behavior (e.g., `#[allow_internal_unstable]`)
///
/// ## Note on Attribute Organization
/// Some attributes like `InlineAttr`, `OptimizeAttr`, and `InstructionSetAttr` are defined separately
/// from this enum because they are used in specific compiler phases (like code generation) and don't
/// need to persist throughout the entire compilation process. They are typically processed and
/// converted into their final form earlier in the compilation pipeline.
///
/// For example:
/// - `InlineAttr` is used during code generation to control function inlining
/// - `OptimizeAttr` is used to control optimization levels
/// - `InstructionSetAttr` is used for target-specific code generation
///
/// These attributes are handled by their respective compiler passes in the [`rustc_codegen_ssa`] crate
/// and don't need to be preserved in the same way as the attributes in this enum.
///
/// For more details on attribute parsing, see the [`rustc_attr_parsing`] crate.
///
/// [`rustc_parse`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/index.html
/// [`rustc_codegen_ssa`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/index.html
/// [`rustc_attr_parsing`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/index.html
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum AttributeKind {
    // tidy-alphabetical-start
    /// Represents `#[align(N)]`.
    Align { align: Align, span: Span },

    /// Represents `#[rustc_allow_const_fn_unstable]`.
    AllowConstFnUnstable(ThinVec<Symbol>),

    /// Represents `#[allow_internal_unstable]`.
    AllowInternalUnstable(ThinVec<(Symbol, Span)>),

    /// Represents `#[rustc_as_ptr]` (used by the `dangling_pointers_from_temporaries` lint).
    AsPtr(Span),

    /// Represents `#[rustc_default_body_unstable]`.
    BodyStability {
        stability: DefaultBodyStability,
        /// Span of the `#[rustc_default_body_unstable(...)]` attribute
        span: Span,
    },

    /// Represents `#[cold]`.
    Cold(Span),

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

    /// Represents `#[inline]` and `#[rustc_force_inline]`.
    Inline(InlineAttr, Span),

    /// Represents `#[rustc_macro_transparency]`.
    MacroTransparency(Transparency),

    /// Represents [`#[may_dangle]`](https://std-dev-guide.rust-lang.org/tricky/may-dangle.html).
    MayDangle(Span),

    /// Represents `#[must_use]`.
    MustUse {
        span: Span,
        /// must_use can optionally have a reason: `#[must_use = "reason this must be used"]`
        reason: Option<Symbol>,
    },

    /// Represents `#[no_mangle]`
    NoMangle(Span),

    /// Represents `#[optimize(size|speed)]`
    Optimize(OptimizeAttr, Span),

    /// Represents `#[rustc_pub_transparent]` (used by the `repr_transparent_external_private_fields` lint).
    PubTransparent(Span),

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
