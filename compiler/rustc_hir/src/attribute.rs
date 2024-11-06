use std::fmt::Display;

use rustc_ast::attr::AttributeExt;
use rustc_ast::token::CommentKind;
use rustc_ast::{self as ast, AttrId, AttrStyle, DelimArgs, MetaItemInner, MetaItemLit};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::hygiene::Transparency;
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol, sym};
use rustc_target::abi::Align;
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::{DefaultBodyStability, ItemLocalId, PartialConstStability, RustcVersion, Stability};

/// The derived implementation of [`HashStable_Generic`] on [`Attribute`]s shouldn't hash
/// [`AttrId`]s. By wrapping them in this, we make sure we never do.
#[derive(Copy, Debug, Encodable, Decodable, Clone)]
pub struct HashIgnoredAttrId {
    pub attr_id: AttrId,
}

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

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum DiagnosticAttribute {
    // tidy-alphabetical-start
    DoNotRecommend,
    OnUnimplemented,
    // tidy-alphabetical-end
}

#[derive(PartialEq, Debug, Encodable, Decodable, Copy, Clone, HashStable_Generic)]
pub enum Repr {
    Int(IntType),
    Rust,
    C,
    Packed(Align),
    Simd,
    Transparent,
    Align(Align),
}

pub enum TransparencyError {
    UnknownTransparency(Symbol, Span),
    MultipleTransparencyAttrs(Span, Span),
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy),
}

impl IntType {
    #[inline]
    pub fn is_signed(self) -> bool {
        use IntType::*;

        match self {
            SignedInt(..) => true,
            UnsignedInt(..) => false,
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

// FIXME(jdonszelmann): guidelines for when an attribute should be "rustc".
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum RustcAttribute {
    // tidy-alphabetical-startt
    AllowIncoherentImpl,
    Coinductive,
    DenyExplicitImpl,
    HasIncoherentImpls,
    LayoutScalarValidRangeEnd,
    LayoutScalarValidRangeStart,
    LintDiagnostics,
    LintOptDenyFieldAccess,
    LintOptTy,
    LintQueryInstability,
    LintUntrackedQueryInformation,
    MustImplementOneOf,
    ObjectLifetimeDefault,
    PassByValue,
    Promotable,
    PubTransparent,
    SafeIntrinsic,
    // tidy-alphabetical-end
}

/// Attributes represent parsed, *built in* attributes. That means,
/// attributes that are not actually ever expanded. They're instead used as markers,
/// to guide the compilation process in various way in most every stage of the compiler.
/// These are kept around after the AST, into the HIR and further on.
///
/// The word parsed could be a little misleading here, because the parser already parses
/// attributes early on. However, the result, an [`ast::Attribute`]
/// is only parsed at a high level, still containing a token stream in many cases. That is
/// because the structure of the contents varies from attribute to attribute.
/// With a parsed attribute I mean that each attribute is processed individually into a
/// final structure, which on-site (the place where the attribute is useful for, think the
/// the place where `must_use` is checked) little to no extra parsing or validating needs to
/// happen.
///
/// For more docs, look in [`rustc_attr`](https://doc.rust-lang.org/stable/nightly-rustc/rustc_attr/index.html)
// FIXME(jdonszelmann): rename to AttributeKind once hir::AttributeKind is dissolved
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum AttributeKind {
    // tidy-alphabetical-start
    Allow,
    AllowConstFnUnstable(ThinVec<Symbol>),
    AllowInternalUnsafe,
    AllowInternalUnstable(ThinVec<Symbol>),
    AutoDiff,
    AutomaticallyDerived,
    BodyStability {
        stability: DefaultBodyStability,
        /// Span of the `#[rustc_default_body_unstable(...)]` attribute
        span: Span,
    },
    Cfg,
    CfgAttr,
    CfiEncoding, // FIXME(cfi_encoding)
    Cold,
    CollapseDebuginfo,
    Confusables {
        symbols: ThinVec<Symbol>,
        // FIXME(jdonszelmann): remove when target validation code is moved
        first_span: Span,
    },
    ConstStabilityIndirect,
    ConstStability {
        stability: PartialConstStability,
        /// Span of the `#[rustc_const_stable(...)]` or `#[rustc_const_unstable(...)]` attribute
        span: Span,
    },
    ConstTrait,
    Coroutine,
    Coverage,
    CustomMir,
    DebuggerVisualizer,
    DefaultLibAllocator,
    Deny,
    DeprecatedSafe, // FIXME(deprecated_safe)
    Deprecation {
        deprecation: Deprecation,
        span: Span,
    },
    Diagnostic(DiagnosticAttribute),
    Doc,
    /// A doc comment (e.g. `/// ...`, `//! ...`, `/** ... */`, `/*! ... */`).
    /// Doc attributes (e.g. `#[doc="..."]`) are represented with the `Normal`
    /// variant (which is much less compact and thus more expensive).
    DocComment {
        style: AttrStyle,
        kind: CommentKind,
        span: Span,
        comment: Symbol,
    },
    Expect,
    ExportName,
    FfiConst,
    FfiPure,
    Forbid,
    Fundamental,
    Ignore,
    // TODO: must contain span for clippy
    Inline,
    InstructionSet, // broken on stable!!!
    Lang,
    Link,
    Linkage,
    LinkName,
    LinkOrdinal,
    LinkSection,
    MacroExport,
    MacroTransparency(Transparency),
    MacroUse,
    Marker,
    MayDangle,
    MustNotSuspend,
    MustUse,
    NeedsAllocator,
    NoImplicitPrelude,
    NoLink,
    NoMangle,
    NonExhaustive,
    NoSanitize,
    OmitGdbPrettyPrinterSection, // FIXME(omit_gdb_pretty_printer_section)
    PanicHandler,
    PatchableFunctionEntry, // FIXME(patchable_function_entry)
    Path,
    Pointee, // FIXME(derive_smart_pointer)
    PreludeImport,
    ProcMacro,
    ProcMacroAttribute,
    ProcMacroDerive,
    Repr(ThinVec<Repr>),
    Rustc(RustcAttribute),
    Stability {
        stability: Stability,
        /// Span of the `#[stable(...)]` or `#[unstable(...)]` attribute
        span: Span,
    },
    Start,
    TargetFeature,
    ThreadLocal,
    TrackCaller,
    Unstable,
    Used,
    Warn,
    WindowsSubsystem, // broken on stable!!!
                      // tidy-alphabetical-end
}

/// Unparsed arguments passed to an attribute macro.
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum AttrArgs {
    /// No arguments: `#[attr]`.
    Empty,
    /// Delimited arguments: `#[attr()/[]/{}]`.
    Delimited(DelimArgs),
    /// Arguments of a key-value attribute: `#[attr = "value"]`.
    Eq {
        /// Span of the `=` token.
        eq_span: Span,
        /// The "value".
        expr: MetaItemLit,
    },
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct AttrPath {
    pub segments: Box<[Ident]>,
    pub span: Span,
}

impl AttrPath {
    pub fn from_ast(path: &ast::Path) -> Self {
        AttrPath {
            segments: path.segments.iter().map(|i| i.ident).collect::<Vec<_>>().into_boxed_slice(),
            span: path.span,
        }
    }
}

impl Display for AttrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("::"))
    }
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct AttrItem {
    // Not lowered to hir::Path because we have no NodeId to resolve to.
    pub path: AttrPath,
    pub args: AttrArgs,

    pub id: HashIgnoredAttrId,
    /// Denotes if the attribute decorates the following construct (outer)
    /// or the construct this attribute is contained within (inner).
    pub style: AttrStyle,
    /// Span of the entire attribute
    pub span: Span,
}

#[derive(Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Attribute {
    /// A parsed built-in attribute.
    ///
    /// Each attribute has a span connected to it. However, you must be somewhat careful using it.
    /// That's because sometimes we merge multiple attributes together, like when an item has
    /// multiple `repr` attributes. In this case the span might not be very useful.
    Parsed(AttributeKind),

    /// An attribute that could not be parsed, out of a token-like representation.
    /// This is the case for custom tool attributes.
    Unparsed(Box<AttrItem>),
}

impl Attribute {
    pub fn get_normal_item(&self) -> &AttrItem {
        match &self {
            Attribute::Unparsed(normal) => &normal,
            _ => panic!("unexpected parsed attribute"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self {
            Attribute::Unparsed(normal) => *normal,
            _ => panic!("unexpected parsed attribute"),
        }
    }

    pub fn value_lit(&self) -> Option<&MetaItemLit> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Eq { eq_span: _, expr }, .. } => Some(expr),
                _ => None,
            },
            _ => None,
        }
    }
}

// All paths through parsed are set to None because AttributeExt is made for parsing.
// i.e. if that's the thing you're still doing on a parsed attribute then you're doing
// something wrong.
// FIXME(jdonszelmann): remove when all attributes are parsed together
impl AttributeExt for Attribute {
    fn id(&self) -> AttrId {
        match &self {
            Attribute::Unparsed(u) => u.id.attr_id,
            _ => panic!(),
        }
    }

    fn meta_item_list(&self) -> Option<ThinVec<ast::MetaItemInner>> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Delimited(d), .. } => {
                    ast::MetaItemKind::list_from_tokens(d.tokens.clone())
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn value_str(&self) -> Option<Symbol> {
        self.value_lit().and_then(|x| x.value_str())
    }

    fn value_span(&self) -> Option<Span> {
        self.value_lit().map(|i| i.span)
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    fn ident(&self) -> Option<Ident> {
        match &self {
            Attribute::Unparsed(n) => {
                if let [ident] = n.path.segments.as_ref() {
                    Some(*ident)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn path_matches(&self, name: &[Symbol]) -> bool {
        match &self {
            Attribute::Unparsed(n) => {
                n.path.segments.len() == name.len()
                    && n.path.segments.iter().zip(name).all(|(s, n)| s.name == *n)
            }
            _ => false,
        }
    }

    fn is_doc_comment(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::DocComment { .. }))
    }

    fn span(&self) -> Span {
        match &self {
            Attribute::Unparsed(u) => u.span,
            _ => panic!(),
        }
    }

    fn is_word(&self) -> bool {
        match &self {
            Attribute::Unparsed(n) => {
                matches!(n.args, AttrArgs::Empty)
            }
            _ => false,
        }
    }

    fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        match &self {
            Attribute::Unparsed(n) => Some(n.path.segments.iter().copied().collect()),
            _ => None,
        }
    }

    fn doc_str(&self) -> Option<Symbol> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { comment, .. }) => Some(*comment),
            Attribute::Unparsed(_) if self.has_name(sym::doc) => self.value_str(),
            _ => None,
        }
    }
    fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { kind, comment, .. }) => {
                Some((*comment, *kind))
            }
            Attribute::Unparsed(_) if self.name_or_empty() == sym::doc => {
                self.value_str().map(|s| (s, CommentKind::Line))
            }
            _ => None,
        }
    }

    fn style(&self) -> AttrStyle {
        match &self {
            Attribute::Unparsed(u) => u.style,
            _ => panic!(),
        }
    }
}

// FIXME(fn_delegation): use function delegation instead of manually forwarding
impl Attribute {
    pub fn id(&self) -> AttrId {
        AttributeExt::id(self)
    }

    pub fn name_or_empty(&self) -> Symbol {
        AttributeExt::name_or_empty(self)
    }

    pub fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        AttributeExt::meta_item_list(self)
    }

    pub fn value_str(&self) -> Option<Symbol> {
        AttributeExt::value_str(self)
    }

    pub fn value_span(&self) -> Option<Span> {
        AttributeExt::value_span(self)
    }

    pub fn ident(&self) -> Option<Ident> {
        AttributeExt::ident(self)
    }

    pub fn path_matches(&self, name: &[Symbol]) -> bool {
        AttributeExt::path_matches(self, name)
    }

    pub fn is_doc_comment(&self) -> bool {
        AttributeExt::is_doc_comment(self)
    }

    #[inline]
    pub fn has_name(&self, name: Symbol) -> bool {
        AttributeExt::has_name(self, name)
    }

    pub fn span(&self) -> Span {
        AttributeExt::span(self)
    }

    pub fn is_word(&self) -> bool {
        AttributeExt::is_word(self)
    }

    pub fn path(&self) -> SmallVec<[Symbol; 1]> {
        AttributeExt::path(self)
    }

    pub fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        AttributeExt::ident_path(self)
    }

    pub fn doc_str(&self) -> Option<Symbol> {
        AttributeExt::doc_str(self)
    }

    pub fn is_proc_macro_attr(&self) -> bool {
        AttributeExt::is_proc_macro_attr(self)
    }

    pub fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        AttributeExt::doc_str_and_comment_kind(self)
    }

    pub fn style(&self) -> AttrStyle {
        AttributeExt::style(self)
    }
}

/// Attributes owned by a HIR owner.
#[derive(Debug)]
pub struct AttributeMap<'tcx> {
    pub map: SortedMap<ItemLocalId, &'tcx [Attribute]>,
    // Only present when the crate hash is needed.
    pub opt_hash: Option<Fingerprint>,
}

// FIXME(jdonszelmann): add more functions here to search through attributes
impl<'tcx> AttributeMap<'tcx> {
    pub const EMPTY: &'static AttributeMap<'static> =
        &AttributeMap { map: SortedMap::new(), opt_hash: Some(Fingerprint::ZERO) };

    #[inline]
    pub fn get(&self, id: ItemLocalId) -> &'tcx [Attribute] {
        self.map.get(&id).copied().unwrap_or(&[])
    }
}
