//! The Rust abstract syntax tree module.
//!
//! This module contains common structures forming the language AST.
//! Two main entities in the module are [`Item`] (which represents an AST element with
//! additional metadata), and [`ItemKind`] (which represents a concrete type and contains
//! information specific to the type of the item).
//!
//! Other module items worth mentioning:
//! - [`Ty`] and [`TyKind`]: A parsed Rust type.
//! - [`Expr`] and [`ExprKind`]: A parsed Rust expression.
//! - [`Pat`] and [`PatKind`]: A parsed Rust pattern. Patterns are often dual to expressions.
//! - [`Stmt`] and [`StmtKind`]: An executable action that does not return a value.
//! - [`FnDecl`], [`FnHeader`] and [`Param`]: Metadata associated with a function declaration.
//! - [`Generics`], [`GenericParam`], [`WhereClause`]: Metadata associated with generic parameters.
//! - [`EnumDef`] and [`Variant`]: Enum declaration.
//! - [`MetaItemLit`] and [`LitKind`]: Literal expressions.
//! - [`MacroDef`], [`MacStmtStyle`], [`MacCall`]: Macro definition and invocation.
//! - [`Attribute`]: Metadata associated with item.
//! - [`UnOp`], [`BinOp`], and [`BinOpKind`]: Unary and binary operators.

pub use crate::format::*;
pub use crate::util::parser::ExprPrecedence;
pub use rustc_span::AttrId;
pub use GenericArgs::*;
pub use UnsafeSource::*;

use crate::ptr::P;
use crate::token::{self, CommentKind, Delimiter};
use crate::tokenstream::{DelimSpan, LazyAttrTokenStream, TokenStream};
pub use rustc_ast_ir::{Movability, Mutability};
use rustc_data_structures::packed::Pu128;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::Lrc;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::source_map::{respan, Spanned};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{ErrorGuaranteed, Span, DUMMY_SP};
use std::borrow::Cow;
use std::cmp;
use std::fmt;
use std::mem;
use thin_vec::{thin_vec, ThinVec};

/// A "Label" is an identifier of some point in sources,
/// e.g. in the following code:
///
/// ```rust
/// 'outer: loop {
///     break 'outer;
/// }
/// ```
///
/// `'outer` is a label.
#[derive(Clone, Encodable, Decodable, Copy, HashStable_Generic, Eq, PartialEq)]
pub struct Label {
    pub ident: Ident,
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label({:?})", self.ident)
    }
}

/// A "Lifetime" is an annotation of the scope in which variable
/// can be used, e.g. `'a` in `&'a i32`.
#[derive(Clone, Encodable, Decodable, Copy, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub id: NodeId,
    pub ident: Ident,
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lifetime({}: {})", self.id, self)
    }
}

impl fmt::Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ident.name)
    }
}

/// A "Path" is essentially Rust's notion of a name.
///
/// It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
///
/// E.g., `std::cmp::PartialEq`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Path {
    pub span: Span,
    /// The segments in the path: the things separated by `::`.
    /// Global paths begin with `kw::PathRoot`.
    pub segments: ThinVec<PathSegment>,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl PartialEq<Symbol> for Path {
    #[inline]
    fn eq(&self, symbol: &Symbol) -> bool {
        self.segments.len() == 1 && { self.segments[0].ident.name == *symbol }
    }
}

impl<CTX: rustc_span::HashStableContext> HashStable<CTX> for Path {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.segments.len().hash_stable(hcx, hasher);
        for segment in &self.segments {
            segment.ident.hash_stable(hcx, hasher);
        }
    }
}

impl Path {
    /// Convert a span and an identifier to the corresponding
    /// one-segment path.
    pub fn from_ident(ident: Ident) -> Path {
        Path { segments: thin_vec![PathSegment::from_ident(ident)], span: ident.span, tokens: None }
    }

    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == kw::PathRoot
    }

    /// If this path is a single identifier with no arguments, does not ensure
    /// that the path resolves to a const param, the caller should check this.
    pub fn is_potential_trivial_const_arg(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].args.is_none()
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of types.
///
/// E.g., `std`, `String` or `Box<T>`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub ident: Ident,

    pub id: NodeId,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`.
    /// `None` means that no parameter list is supplied (`Path`),
    /// `Some` means that parameter list is supplied (`Path<X, Y>`)
    /// but it can be empty (`Path<>`).
    /// `P` is used as a size optimization for the common case with no parameters.
    pub args: Option<P<GenericArgs>>,
}

impl PathSegment {
    pub fn from_ident(ident: Ident) -> Self {
        PathSegment { ident, id: DUMMY_NODE_ID, args: None }
    }

    pub fn path_root(span: Span) -> Self {
        PathSegment::from_ident(Ident::new(kw::PathRoot, span))
    }

    pub fn span(&self) -> Span {
        match &self.args {
            Some(args) => self.ident.span.to(args.span()),
            None => self.ident.span,
        }
    }
}

/// The generic arguments and associated item constraints of a path segment.
///
/// E.g., `<A, B>` as in `Foo<A, B>` or `(A, B)` as in `Foo(A, B)`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericArgs {
    /// The `<'a, A, B, C>` in `foo::bar::baz::<'a, A, B, C>`.
    AngleBracketed(AngleBracketedArgs),
    /// The `(A, B)` and `C` in `Foo(A, B) -> C`.
    Parenthesized(ParenthesizedArgs),
    /// `(..)` in return type notation.
    ParenthesizedElided(Span),
}

impl GenericArgs {
    pub fn is_angle_bracketed(&self) -> bool {
        matches!(self, AngleBracketed(..))
    }

    pub fn span(&self) -> Span {
        match self {
            AngleBracketed(data) => data.span,
            Parenthesized(data) => data.span,
            ParenthesizedElided(span) => *span,
        }
    }
}

/// Concrete argument in the sequence of generic args.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericArg {
    /// `'a` in `Foo<'a>`.
    Lifetime(Lifetime),
    /// `Bar` in `Foo<Bar>`.
    Type(P<Ty>),
    /// `1` in `Foo<1>`.
    Const(AnonConst),
}

impl GenericArg {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(lt) => lt.ident.span,
            GenericArg::Type(ty) => ty.span,
            GenericArg::Const(ct) => ct.value.span,
        }
    }
}

/// A path like `Foo<'a, T>`.
#[derive(Clone, Encodable, Decodable, Debug, Default)]
pub struct AngleBracketedArgs {
    /// The overall span.
    pub span: Span,
    /// The comma separated parts in the `<...>`.
    pub args: ThinVec<AngleBracketedArg>,
}

/// Either an argument for a generic parameter or a constraint on an associated item.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AngleBracketedArg {
    /// A generic argument for a generic parameter.
    Arg(GenericArg),
    /// A constraint on an associated item.
    Constraint(AssocItemConstraint),
}

impl AngleBracketedArg {
    pub fn span(&self) -> Span {
        match self {
            AngleBracketedArg::Arg(arg) => arg.span(),
            AngleBracketedArg::Constraint(constraint) => constraint.span,
        }
    }
}

impl Into<P<GenericArgs>> for AngleBracketedArgs {
    fn into(self) -> P<GenericArgs> {
        P(GenericArgs::AngleBracketed(self))
    }
}

impl Into<P<GenericArgs>> for ParenthesizedArgs {
    fn into(self) -> P<GenericArgs> {
        P(GenericArgs::Parenthesized(self))
    }
}

/// A path like `Foo(A, B) -> C`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ParenthesizedArgs {
    /// ```text
    /// Foo(A, B) -> C
    /// ^^^^^^^^^^^^^^
    /// ```
    pub span: Span,

    /// `(A, B)`
    pub inputs: ThinVec<P<Ty>>,

    /// ```text
    /// Foo(A, B) -> C
    ///    ^^^^^^
    /// ```
    pub inputs_span: Span,

    /// `C`
    pub output: FnRetTy,
}

impl ParenthesizedArgs {
    pub fn as_angle_bracketed_args(&self) -> AngleBracketedArgs {
        let args = self
            .inputs
            .iter()
            .cloned()
            .map(|input| AngleBracketedArg::Arg(GenericArg::Type(input)))
            .collect();
        AngleBracketedArgs { span: self.inputs_span, args }
    }
}

pub use crate::node_id::{NodeId, CRATE_NODE_ID, DUMMY_NODE_ID};

/// Modifiers on a trait bound like `~const`, `?` and `!`.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug)]
pub struct TraitBoundModifiers {
    pub constness: BoundConstness,
    pub asyncness: BoundAsyncness,
    pub polarity: BoundPolarity,
}

impl TraitBoundModifiers {
    pub const NONE: Self = Self {
        constness: BoundConstness::Never,
        asyncness: BoundAsyncness::Normal,
        polarity: BoundPolarity::Positive,
    };
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericBound {
    Trait(PolyTraitRef, TraitBoundModifiers),
    Outlives(Lifetime),
    /// Precise capturing syntax: `impl Sized + use<'a>`
    Use(ThinVec<PreciseCapturingArg>, Span),
}

impl GenericBound {
    pub fn span(&self) -> Span {
        match self {
            GenericBound::Trait(t, ..) => t.span,
            GenericBound::Outlives(l) => l.ident.span,
            GenericBound::Use(_, span) => *span,
        }
    }
}

pub type GenericBounds = Vec<GenericBound>;

/// Specifies the enforced ordering for generic parameters. In the future,
/// if we wanted to relax this order, we could override `PartialEq` and
/// `PartialOrd`, to allow the kinds to be unordered.
#[derive(Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParamKindOrd {
    Lifetime,
    TypeOrConst,
}

impl fmt::Display for ParamKindOrd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamKindOrd::Lifetime => "lifetime".fmt(f),
            ParamKindOrd::TypeOrConst => "type and const".fmt(f),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericParamKind {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime,
    Type {
        default: Option<P<Ty>>,
    },
    Const {
        ty: P<Ty>,
        /// Span of the `const` keyword.
        kw_span: Span,
        /// Optional default value for the const generic param.
        default: Option<AnonConst>,
    },
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct GenericParam {
    pub id: NodeId,
    pub ident: Ident,
    pub attrs: AttrVec,
    pub bounds: GenericBounds,
    pub is_placeholder: bool,
    pub kind: GenericParamKind,
    pub colon_span: Option<Span>,
}

impl GenericParam {
    pub fn span(&self) -> Span {
        match &self.kind {
            GenericParamKind::Lifetime | GenericParamKind::Type { default: None } => {
                self.ident.span
            }
            GenericParamKind::Type { default: Some(ty) } => self.ident.span.to(ty.span),
            GenericParamKind::Const { kw_span, default: Some(default), .. } => {
                kw_span.to(default.value.span)
            }
            GenericParamKind::Const { kw_span, default: None, ty } => kw_span.to(ty.span),
        }
    }
}

/// Represents lifetime, type and const parameters attached to a declaration of
/// a function, enum, trait, etc.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Generics {
    pub params: ThinVec<GenericParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Default for Generics {
    /// Creates an instance of `Generics`.
    fn default() -> Generics {
        Generics { params: ThinVec::new(), where_clause: Default::default(), span: DUMMY_SP }
    }
}

/// A where-clause in a definition.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereClause {
    /// `true` if we ate a `where` token.
    ///
    /// This can happen if we parsed no predicates, e.g., `struct Foo where {}`.
    /// This allows us to pretty-print accurately and provide correct suggestion diagnostics.
    pub has_where_token: bool,
    pub predicates: ThinVec<WherePredicate>,
    pub span: Span,
}

impl Default for WhereClause {
    fn default() -> WhereClause {
        WhereClause { has_where_token: false, predicates: ThinVec::new(), span: DUMMY_SP }
    }
}

/// A single predicate in a where-clause.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum WherePredicate {
    /// A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
    BoundPredicate(WhereBoundPredicate),
    /// A lifetime predicate (e.g., `'a: 'b + 'c`).
    RegionPredicate(WhereRegionPredicate),
    /// An equality predicate (unsupported).
    EqPredicate(WhereEqPredicate),
}

impl WherePredicate {
    pub fn span(&self) -> Span {
        match self {
            WherePredicate::BoundPredicate(p) => p.span,
            WherePredicate::RegionPredicate(p) => p.span,
            WherePredicate::EqPredicate(p) => p.span,
        }
    }
}

/// A type bound.
///
/// E.g., `for<'c> Foo: Send + Clone + 'c`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any generics from a `for` binding.
    pub bound_generic_params: ThinVec<GenericParam>,
    /// The type being bounded.
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (`Clone + Send + 'static`).
    pub bounds: GenericBounds,
}

/// A lifetime predicate.
///
/// E.g., `'a: 'b + 'c`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds,
}

/// An equality predicate (unsupported).
///
/// E.g., `T = int`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereEqPredicate {
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Crate {
    pub attrs: AttrVec,
    pub items: ThinVec<P<Item>>,
    pub spans: ModSpans,
    /// Must be equal to `CRATE_NODE_ID` after the crate root is expanded, but may hold
    /// expansion placeholders or an unassigned value (`DUMMY_NODE_ID`) before that.
    pub id: NodeId,
    pub is_placeholder: bool,
}

/// A semantic representation of a meta item. A meta item is a slightly
/// restricted form of an attribute -- it can only contain expressions in
/// certain leaf positions, rather than arbitrary token streams -- that is used
/// for most built-in attributes.
///
/// E.g., `#[test]`, `#[derive(..)]`, `#[rustfmt::skip]` or `#[feature = "foo"]`.
#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MetaItem {
    pub unsafety: Safety,
    pub path: Path,
    pub kind: MetaItemKind,
    pub span: Span,
}

/// The meta item kind, containing the data after the initial path.
#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum MetaItemKind {
    /// Word meta item.
    ///
    /// E.g., `#[test]`, which lacks any arguments after `test`.
    Word,

    /// List meta item.
    ///
    /// E.g., `#[derive(..)]`, where the field represents the `..`.
    List(ThinVec<NestedMetaItem>),

    /// Name value meta item.
    ///
    /// E.g., `#[feature = "foo"]`, where the field represents the `"foo"`.
    NameValue(MetaItemLit),
}

/// Values inside meta item lists.
///
/// E.g., each of `Clone`, `Copy` in `#[derive(Clone, Copy)]`.
#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum NestedMetaItem {
    /// A full MetaItem, for recursive meta items.
    MetaItem(MetaItem),

    /// A literal.
    ///
    /// E.g., `"foo"`, `64`, `true`.
    Lit(MetaItemLit),
}

/// A block (`{ .. }`).
///
/// E.g., `{ .. }` as in `fn foo() { .. }`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Block {
    /// The statements in the block.
    pub stmts: ThinVec<Stmt>,
    pub id: NodeId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`.
    pub rules: BlockCheckMode,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
    /// The following *isn't* a parse error, but will cause multiple errors in following stages.
    /// ```compile_fail
    /// let x = {
    ///     foo: var
    /// };
    /// ```
    /// #34255
    pub could_be_bare_literal: bool,
}

/// A match pattern.
///
/// Patterns appear in match statements and some other contexts, such as `let` and `if let`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Pat {
    pub id: NodeId,
    pub kind: PatKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Pat {
    /// Attempt reparsing the pattern as a type.
    /// This is intended for use by diagnostics.
    pub fn to_ty(&self) -> Option<P<Ty>> {
        let kind = match &self.kind {
            // In a type expression `_` is an inference variable.
            PatKind::Wild => TyKind::Infer,
            // An IDENT pattern with no binding mode would be valid as path to a type. E.g. `u32`.
            PatKind::Ident(BindingMode::NONE, ident, None) => {
                TyKind::Path(None, Path::from_ident(*ident))
            }
            PatKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            PatKind::MacCall(mac) => TyKind::MacCall(mac.clone()),
            // `&mut? P` can be reinterpreted as `&mut? T` where `T` is `P` reparsed as a type.
            PatKind::Ref(pat, mutbl) => {
                pat.to_ty().map(|ty| TyKind::Ref(None, MutTy { ty, mutbl: *mutbl }))?
            }
            // A slice/array pattern `[P]` can be reparsed as `[T]`, an unsized array,
            // when `P` can be reparsed as a type `T`.
            PatKind::Slice(pats) if pats.len() == 1 => pats[0].to_ty().map(TyKind::Slice)?,
            // A tuple pattern `(P0, .., Pn)` can be reparsed as `(T0, .., Tn)`
            // assuming `T0` to `Tn` are all syntactically valid as types.
            PatKind::Tuple(pats) => {
                let mut tys = ThinVec::with_capacity(pats.len());
                // FIXME(#48994) - could just be collected into an Option<Vec>
                for pat in pats {
                    tys.push(pat.to_ty()?);
                }
                TyKind::Tup(tys)
            }
            _ => return None,
        };

        Some(P(Ty { kind, id: self.id, span: self.span, tokens: None }))
    }

    /// Walk top-down and call `it` in each place where a pattern occurs
    /// starting with the root pattern `walk` is called on. If `it` returns
    /// false then we will descend no further but siblings will be processed.
    pub fn walk(&self, it: &mut impl FnMut(&Pat) -> bool) {
        if !it(self) {
            return;
        }

        match &self.kind {
            // Walk into the pattern associated with `Ident` (if any).
            PatKind::Ident(_, _, Some(p)) => p.walk(it),

            // Walk into each field of struct.
            PatKind::Struct(_, _, fields, _) => fields.iter().for_each(|field| field.pat.walk(it)),

            // Sequence of patterns.
            PatKind::TupleStruct(_, _, s)
            | PatKind::Tuple(s)
            | PatKind::Slice(s)
            | PatKind::Or(s) => s.iter().for_each(|p| p.walk(it)),

            // Trivial wrappers over inner patterns.
            PatKind::Box(s) | PatKind::Deref(s) | PatKind::Ref(s, _) | PatKind::Paren(s) => {
                s.walk(it)
            }

            // These patterns do not contain subpatterns, skip.
            PatKind::Wild
            | PatKind::Rest
            | PatKind::Never
            | PatKind::Lit(_)
            | PatKind::Range(..)
            | PatKind::Ident(..)
            | PatKind::Path(..)
            | PatKind::MacCall(_)
            | PatKind::Err(_) => {}
        }
    }

    /// Is this a `..` pattern?
    pub fn is_rest(&self) -> bool {
        matches!(self.kind, PatKind::Rest)
    }

    /// Whether this could be a never pattern, taking into account that a macro invocation can
    /// return a never pattern. Used to inform errors during parsing.
    pub fn could_be_never_pattern(&self) -> bool {
        let mut could_be_never_pattern = false;
        self.walk(&mut |pat| match &pat.kind {
            PatKind::Never | PatKind::MacCall(_) => {
                could_be_never_pattern = true;
                false
            }
            PatKind::Or(s) => {
                could_be_never_pattern = s.iter().all(|p| p.could_be_never_pattern());
                false
            }
            _ => true,
        });
        could_be_never_pattern
    }

    /// Whether this contains a `!` pattern. This in particular means that a feature gate error will
    /// be raised if the feature is off. Used to avoid gating the feature twice.
    pub fn contains_never_pattern(&self) -> bool {
        let mut contains_never_pattern = false;
        self.walk(&mut |pat| {
            if matches!(pat.kind, PatKind::Never) {
                contains_never_pattern = true;
            }
            true
        });
        contains_never_pattern
    }

    /// Return a name suitable for diagnostics.
    pub fn descr(&self) -> Option<String> {
        match &self.kind {
            PatKind::Wild => Some("_".to_string()),
            PatKind::Ident(BindingMode::NONE, ident, None) => Some(format!("{ident}")),
            PatKind::Ref(pat, mutbl) => pat.descr().map(|d| format!("&{}{d}", mutbl.prefix_str())),
            _ => None,
        }
    }
}

/// A single field in a struct pattern.
///
/// Patterns like the fields of `Foo { x, ref y, ref mut z }`
/// are treated the same as `x: x, y: ref y, z: ref mut z`,
/// except when `is_shorthand` is true.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PatField {
    /// The identifier for the field.
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: P<Pat>,
    pub is_shorthand: bool,
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub is_placeholder: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum ByRef {
    Yes(Mutability),
    No,
}

impl ByRef {
    #[must_use]
    pub fn cap_ref_mutability(mut self, mutbl: Mutability) -> Self {
        if let ByRef::Yes(old_mutbl) = &mut self {
            *old_mutbl = cmp::min(*old_mutbl, mutbl);
        }
        self
    }
}

/// The mode of a binding (`mut`, `ref mut`, etc).
/// Used for both the explicit binding annotations given in the HIR for a binding
/// and the final binding mode that we infer after type inference/match ergonomics.
/// `.0` is the by-reference mode (`ref`, `ref mut`, or by value),
/// `.1` is the mutability of the binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub struct BindingMode(pub ByRef, pub Mutability);

impl BindingMode {
    pub const NONE: Self = Self(ByRef::No, Mutability::Not);
    pub const REF: Self = Self(ByRef::Yes(Mutability::Not), Mutability::Not);
    pub const MUT: Self = Self(ByRef::No, Mutability::Mut);
    pub const REF_MUT: Self = Self(ByRef::Yes(Mutability::Mut), Mutability::Not);
    pub const MUT_REF: Self = Self(ByRef::Yes(Mutability::Not), Mutability::Mut);
    pub const MUT_REF_MUT: Self = Self(ByRef::Yes(Mutability::Mut), Mutability::Mut);

    pub fn prefix_str(self) -> &'static str {
        match self {
            Self::NONE => "",
            Self::REF => "ref ",
            Self::MUT => "mut ",
            Self::REF_MUT => "ref mut ",
            Self::MUT_REF => "mut ref ",
            Self::MUT_REF_MUT => "mut ref mut ",
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum RangeEnd {
    /// `..=` or `...`
    Included(RangeSyntax),
    /// `..`
    Excluded,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum RangeSyntax {
    /// `...`
    DotDotDot,
    /// `..=`
    DotDotEq,
}

/// All the different flavors of pattern that Rust recognizes.
//
// Adding a new variant? Please update `test_pat` in `tests/ui/macros/stringify.rs`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum PatKind {
    /// Represents a wildcard pattern (`_`).
    Wild,

    /// A `PatKind::Ident` may either be a new bound variable (`ref mut binding @ OPT_SUBPATTERN`),
    /// or a unit struct/variant pattern, or a const pattern (in the last two cases the third
    /// field must be `None`). Disambiguation cannot be done with parser alone, so it happens
    /// during name resolution.
    Ident(BindingMode, Ident, Option<P<Pat>>),

    /// A struct or struct variant pattern (e.g., `Variant {x, y, ..}`).
    Struct(Option<P<QSelf>>, Path, ThinVec<PatField>, PatFieldsRest),

    /// A tuple struct/variant pattern (`Variant(x, y, .., z)`).
    TupleStruct(Option<P<QSelf>>, Path, ThinVec<P<Pat>>),

    /// An or-pattern `A | B | C`.
    /// Invariant: `pats.len() >= 2`.
    Or(ThinVec<P<Pat>>),

    /// A possibly qualified path pattern.
    /// Unqualified path patterns `A::B::C` can legally refer to variants, structs, constants
    /// or associated constants. Qualified path patterns `<A>::B::C`/`<A as Trait>::B::C` can
    /// only legally refer to associated constants.
    Path(Option<P<QSelf>>, Path),

    /// A tuple pattern (`(a, b)`).
    Tuple(ThinVec<P<Pat>>),

    /// A `box` pattern.
    Box(P<Pat>),

    /// A `deref` pattern (currently `deref!()` macro-based syntax).
    Deref(P<Pat>),

    /// A reference pattern (e.g., `&mut (a, b)`).
    Ref(P<Pat>, Mutability),

    /// A literal.
    Lit(P<Expr>),

    /// A range pattern (e.g., `1...2`, `1..2`, `1..`, `..2`, `1..=2`, `..=2`).
    Range(Option<P<Expr>>, Option<P<Expr>>, Spanned<RangeEnd>),

    /// A slice pattern `[a, b, c]`.
    Slice(ThinVec<P<Pat>>),

    /// A rest pattern `..`.
    ///
    /// Syntactically it is valid anywhere.
    ///
    /// Semantically however, it only has meaning immediately inside:
    /// - a slice pattern: `[a, .., b]`,
    /// - a binding pattern immediately inside a slice pattern: `[a, r @ ..]`,
    /// - a tuple pattern: `(a, .., b)`,
    /// - a tuple struct/variant pattern: `$path(a, .., b)`.
    ///
    /// In all of these cases, an additional restriction applies,
    /// only one rest pattern may occur in the pattern sequences.
    Rest,

    // A never pattern `!`.
    Never,

    /// Parentheses in patterns used for grouping (i.e., `(PAT)`).
    Paren(P<Pat>),

    /// A macro pattern; pre-expansion.
    MacCall(P<MacCall>),

    /// Placeholder for a pattern that wasn't syntactically well formed in some way.
    Err(ErrorGuaranteed),
}

/// Whether the `..` is present in a struct fields pattern.
#[derive(Clone, Copy, Encodable, Decodable, Debug, PartialEq)]
pub enum PatFieldsRest {
    /// `module::StructName { field, ..}`
    Rest,
    /// `module::StructName { field }`
    None,
}

/// The kind of borrow in an `AddrOf` expression,
/// e.g., `&place` or `&raw const place`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum BorrowKind {
    /// A normal borrow, `&$expr` or `&mut $expr`.
    /// The resulting type is either `&'a T` or `&'a mut T`
    /// where `T = typeof($expr)` and `'a` is some lifetime.
    Ref,
    /// A raw borrow, `&raw const $expr` or `&raw mut $expr`.
    /// The resulting type is either `*const T` or `*mut T`
    /// where `T = typeof($expr)`.
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum BinOpKind {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `&&` operator (logical and)
    And,
    /// The `||` operator (logical or)
    Or,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

impl BinOpKind {
    pub fn as_str(&self) -> &'static str {
        use BinOpKind::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Rem => "%",
            And => "&&",
            Or => "||",
            BitXor => "^",
            BitAnd => "&",
            BitOr => "|",
            Shl => "<<",
            Shr => ">>",
            Eq => "==",
            Lt => "<",
            Le => "<=",
            Ne => "!=",
            Ge => ">=",
            Gt => ">",
        }
    }

    pub fn is_lazy(&self) -> bool {
        matches!(self, BinOpKind::And | BinOpKind::Or)
    }

    pub fn is_comparison(self) -> bool {
        crate::util::parser::AssocOp::from_ast_binop(self).is_comparison()
    }

    /// Returns `true` if the binary operator takes its arguments by value.
    pub fn is_by_value(self) -> bool {
        !self.is_comparison()
    }
}

pub type BinOp = Spanned<BinOpKind>;

/// Unary operator.
///
/// Note that `&data` is not an operator, it's an `AddrOf` expression.
#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum UnOp {
    /// The `*` operator for dereferencing
    Deref,
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl UnOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            UnOp::Deref => "*",
            UnOp::Not => "!",
            UnOp::Neg => "-",
        }
    }

    /// Returns `true` if the unary operator takes its argument by value.
    pub fn is_by_value(self) -> bool {
        matches!(self, Self::Neg | Self::Not)
    }
}

/// A statement. No `attrs` or `tokens` fields because each `StmtKind` variant
/// contains an AST node with those fields. (Except for `StmtKind::Empty`,
/// which never has attrs or tokens)
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Stmt {
    pub id: NodeId,
    pub kind: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn has_trailing_semicolon(&self) -> bool {
        match &self.kind {
            StmtKind::Semi(_) => true,
            StmtKind::MacCall(mac) => matches!(mac.style, MacStmtStyle::Semicolon),
            _ => false,
        }
    }

    /// Converts a parsed `Stmt` to a `Stmt` with
    /// a trailing semicolon.
    ///
    /// This only modifies the parsed AST struct, not the attached
    /// `LazyAttrTokenStream`. The parser is responsible for calling
    /// `ToAttrTokenStream::add_trailing_semi` when there is actually
    /// a semicolon in the tokenstream.
    pub fn add_trailing_semicolon(mut self) -> Self {
        self.kind = match self.kind {
            StmtKind::Expr(expr) => StmtKind::Semi(expr),
            StmtKind::MacCall(mac) => {
                StmtKind::MacCall(mac.map(|MacCallStmt { mac, style: _, attrs, tokens }| {
                    MacCallStmt { mac, style: MacStmtStyle::Semicolon, attrs, tokens }
                }))
            }
            kind => kind,
        };

        self
    }

    pub fn is_item(&self) -> bool {
        matches!(self.kind, StmtKind::Item(_))
    }

    pub fn is_expr(&self) -> bool {
        matches!(self.kind, StmtKind::Expr(_))
    }
}

// Adding a new variant? Please update `test_stmt` in `tests/ui/macros/stringify.rs`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum StmtKind {
    /// A local (let) binding.
    Let(P<Local>),
    /// An item definition.
    Item(P<Item>),
    /// Expr without trailing semi-colon.
    Expr(P<Expr>),
    /// Expr with a trailing semi-colon.
    Semi(P<Expr>),
    /// Just a trailing semi-colon.
    Empty,
    /// Macro.
    MacCall(P<MacCallStmt>),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MacCallStmt {
    pub mac: P<MacCall>,
    pub style: MacStmtStyle,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug)]
pub enum MacStmtStyle {
    /// The macro statement had a trailing semicolon (e.g., `foo! { ... };`
    /// `foo!(...);`, `foo![...];`).
    Semicolon,
    /// The macro statement had braces (e.g., `foo! { ... }`).
    Braces,
    /// The macro statement had parentheses or brackets and no semicolon (e.g.,
    /// `foo!(...)`). All of these will end up being converted into macro
    /// expressions.
    NoBraces,
}

/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Local {
    pub id: NodeId,
    pub pat: P<Pat>,
    pub ty: Option<P<Ty>>,
    pub kind: LocalKind,
    pub span: Span,
    pub colon_sp: Option<Span>,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum LocalKind {
    /// Local declaration.
    /// Example: `let x;`
    Decl,
    /// Local declaration with an initializer.
    /// Example: `let x = y;`
    Init(P<Expr>),
    /// Local declaration with an initializer and an `else` clause.
    /// Example: `let Some(x) = y else { return };`
    InitElse(P<Expr>, P<Block>),
}

impl LocalKind {
    pub fn init(&self) -> Option<&Expr> {
        match self {
            Self::Decl => None,
            Self::Init(i) | Self::InitElse(i, _) => Some(i),
        }
    }

    pub fn init_else_opt(&self) -> Option<(&Expr, Option<&Block>)> {
        match self {
            Self::Decl => None,
            Self::Init(init) => Some((init, None)),
            Self::InitElse(init, els) => Some((init, Some(els))),
        }
    }
}

/// An arm of a 'match'.
///
/// E.g., `0..=10 => { println!("match!") }` as in
///
/// ```
/// match 123 {
///     0..=10 => { println!("match!") },
///     _ => { println!("no match!") },
/// }
/// ```
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Arm {
    pub attrs: AttrVec,
    /// Match arm pattern, e.g. `10` in `match foo { 10 => {}, _ => {} }`.
    pub pat: P<Pat>,
    /// Match arm guard, e.g. `n > 10` in `match foo { n if n > 10 => {}, _ => {} }`.
    pub guard: Option<P<Expr>>,
    /// Match arm body. Omitted if the pattern is a never pattern.
    pub body: Option<P<Expr>>,
    pub span: Span,
    pub id: NodeId,
    pub is_placeholder: bool,
}

/// A single field in a struct expression, e.g. `x: value` and `y` in `Foo { x: value, y }`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ExprField {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub ident: Ident,
    pub expr: P<Expr>,
    pub is_shorthand: bool,
    pub is_placeholder: bool,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
pub enum BlockCheckMode {
    Default,
    Unsafe(UnsafeSource),
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct AnonConst {
    pub id: NodeId,
    pub value: P<Expr>,
}

/// An expression.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    pub span: Span,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Expr {
    /// Is this expr either `N`, or `{ N }`.
    ///
    /// If this is not the case, name resolution does not resolve `N` when using
    /// `min_const_generics` as more complex expressions are not supported.
    ///
    /// Does not ensure that the path resolves to a const param, the caller should check this.
    pub fn is_potential_trivial_const_arg(&self) -> bool {
        let this = if let ExprKind::Block(block, None) = &self.kind
            && block.stmts.len() == 1
            && let StmtKind::Expr(expr) = &block.stmts[0].kind
        {
            expr
        } else {
            self
        };

        if let ExprKind::Path(None, path) = &this.kind
            && path.is_potential_trivial_const_arg()
        {
            true
        } else {
            false
        }
    }

    pub fn to_bound(&self) -> Option<GenericBound> {
        match &self.kind {
            ExprKind::Path(None, path) => Some(GenericBound::Trait(
                PolyTraitRef::new(ThinVec::new(), path.clone(), self.span),
                TraitBoundModifiers::NONE,
            )),
            _ => None,
        }
    }

    pub fn peel_parens(&self) -> &Expr {
        let mut expr = self;
        while let ExprKind::Paren(inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn peel_parens_and_refs(&self) -> &Expr {
        let mut expr = self;
        while let ExprKind::Paren(inner) | ExprKind::AddrOf(BorrowKind::Ref, _, inner) = &expr.kind
        {
            expr = inner;
        }
        expr
    }

    /// Attempts to reparse as `Ty` (for diagnostic purposes).
    pub fn to_ty(&self) -> Option<P<Ty>> {
        let kind = match &self.kind {
            // Trivial conversions.
            ExprKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            ExprKind::MacCall(mac) => TyKind::MacCall(mac.clone()),

            ExprKind::Paren(expr) => expr.to_ty().map(TyKind::Paren)?,

            ExprKind::AddrOf(BorrowKind::Ref, mutbl, expr) => {
                expr.to_ty().map(|ty| TyKind::Ref(None, MutTy { ty, mutbl: *mutbl }))?
            }

            ExprKind::Repeat(expr, expr_len) => {
                expr.to_ty().map(|ty| TyKind::Array(ty, expr_len.clone()))?
            }

            ExprKind::Array(exprs) if exprs.len() == 1 => exprs[0].to_ty().map(TyKind::Slice)?,

            ExprKind::Tup(exprs) => {
                let tys = exprs.iter().map(|expr| expr.to_ty()).collect::<Option<ThinVec<_>>>()?;
                TyKind::Tup(tys)
            }

            // If binary operator is `Add` and both `lhs` and `rhs` are trait bounds,
            // then type of result is trait object.
            // Otherwise we don't assume the result type.
            ExprKind::Binary(binop, lhs, rhs) if binop.node == BinOpKind::Add => {
                if let (Some(lhs), Some(rhs)) = (lhs.to_bound(), rhs.to_bound()) {
                    TyKind::TraitObject(vec![lhs, rhs], TraitObjectSyntax::None)
                } else {
                    return None;
                }
            }

            ExprKind::Underscore => TyKind::Infer,

            // This expression doesn't look like a type syntactically.
            _ => return None,
        };

        Some(P(Ty { kind, id: self.id, span: self.span, tokens: None }))
    }

    pub fn precedence(&self) -> ExprPrecedence {
        match self.kind {
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::ConstBlock(_) => ExprPrecedence::ConstBlock,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) | ExprKind::IncludedBytes(..) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::Let(..) => ExprPrecedence::Let,
            ExprKind::If(..) => ExprPrecedence::If,
            ExprKind::While(..) => ExprPrecedence::While,
            ExprKind::ForLoop { .. } => ExprPrecedence::ForLoop,
            ExprKind::Loop(..) => ExprPrecedence::Loop,
            ExprKind::Match(_, _, MatchKind::Prefix) => ExprPrecedence::Match,
            ExprKind::Match(_, _, MatchKind::Postfix) => ExprPrecedence::PostfixMatch,
            ExprKind::Closure(..) => ExprPrecedence::Closure,
            ExprKind::Block(..) => ExprPrecedence::Block,
            ExprKind::TryBlock(..) => ExprPrecedence::TryBlock,
            ExprKind::Gen(..) => ExprPrecedence::Gen,
            ExprKind::Await(..) => ExprPrecedence::Await,
            ExprKind::Assign(..) => ExprPrecedence::Assign,
            ExprKind::AssignOp(..) => ExprPrecedence::AssignOp,
            ExprKind::Field(..) => ExprPrecedence::Field,
            ExprKind::Index(..) => ExprPrecedence::Index,
            ExprKind::Range(..) => ExprPrecedence::Range,
            ExprKind::Underscore => ExprPrecedence::Path,
            ExprKind::Path(..) => ExprPrecedence::Path,
            ExprKind::AddrOf(..) => ExprPrecedence::AddrOf,
            ExprKind::Break(..) => ExprPrecedence::Break,
            ExprKind::Continue(..) => ExprPrecedence::Continue,
            ExprKind::Ret(..) => ExprPrecedence::Ret,
            ExprKind::InlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::OffsetOf(..) => ExprPrecedence::OffsetOf,
            ExprKind::MacCall(..) => ExprPrecedence::Mac,
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Paren(..) => ExprPrecedence::Paren,
            ExprKind::Try(..) => ExprPrecedence::Try,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
            ExprKind::Yeet(..) => ExprPrecedence::Yeet,
            ExprKind::FormatArgs(..) => ExprPrecedence::FormatArgs,
            ExprKind::Become(..) => ExprPrecedence::Become,
            ExprKind::Err(_) | ExprKind::Dummy => ExprPrecedence::Err,
        }
    }

    /// To a first-order approximation, is this a pattern?
    pub fn is_approximately_pattern(&self) -> bool {
        matches!(
            &self.peel_parens().kind,
            ExprKind::Array(_)
                | ExprKind::Call(_, _)
                | ExprKind::Tup(_)
                | ExprKind::Lit(_)
                | ExprKind::Range(_, _, _)
                | ExprKind::Underscore
                | ExprKind::Path(_, _)
                | ExprKind::Struct(_)
        )
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Closure {
    pub binder: ClosureBinder,
    pub capture_clause: CaptureBy,
    pub constness: Const,
    pub coroutine_kind: Option<CoroutineKind>,
    pub movability: Movability,
    pub fn_decl: P<FnDecl>,
    pub body: P<Expr>,
    /// The span of the declaration block: 'move |...| -> ...'
    pub fn_decl_span: Span,
    /// The span of the argument block `|...|`
    pub fn_arg_span: Span,
}

/// Limit types of a range (inclusive or exclusive).
#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug)]
pub enum RangeLimits {
    /// Inclusive at the beginning, exclusive at the end.
    HalfOpen,
    /// Inclusive at the beginning and end.
    Closed,
}

/// A method call (e.g. `x.foo::<Bar, Baz>(a, b, c)`).
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MethodCall {
    /// The method name and its generic arguments, e.g. `foo::<Bar, Baz>`.
    pub seg: PathSegment,
    /// The receiver, e.g. `x`.
    pub receiver: P<Expr>,
    /// The arguments, e.g. `a, b, c`.
    pub args: ThinVec<P<Expr>>,
    /// The span of the function, without the dot and receiver e.g. `foo::<Bar,
    /// Baz>(a, b, c)`.
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum StructRest {
    /// `..x`.
    Base(P<Expr>),
    /// `..`.
    Rest(Span),
    /// No trailing `..` or expression.
    None,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct StructExpr {
    pub qself: Option<P<QSelf>>,
    pub path: Path,
    pub fields: ThinVec<ExprField>,
    pub rest: StructRest,
}

// Adding a new variant? Please update `test_expr` in `tests/ui/macros/stringify.rs`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ExprKind {
    /// An array (e.g, `[a, b, c, d]`).
    Array(ThinVec<P<Expr>>),
    /// Allow anonymous constants from an inline `const` block.
    ConstBlock(AnonConst),
    /// A function call.
    ///
    /// The first field resolves to the function itself,
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(P<Expr>, ThinVec<P<Expr>>),
    /// A method call (e.g., `x.foo::<Bar, Baz>(a, b, c)`).
    MethodCall(Box<MethodCall>),
    /// A tuple (e.g., `(a, b, c, d)`).
    Tup(ThinVec<P<Expr>>),
    /// A binary operation (e.g., `a + b`, `a * b`).
    Binary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (e.g., `!x`, `*x`).
    Unary(UnOp, P<Expr>),
    /// A literal (e.g., `1`, `"foo"`).
    Lit(token::Lit),
    /// A cast (e.g., `foo as f64`).
    Cast(P<Expr>, P<Ty>),
    /// A type ascription (e.g., `builtin # type_ascribe(42, usize)`).
    ///
    /// Usually not written directly in user code but
    /// indirectly via the macro `type_ascribe!(...)`.
    Type(P<Expr>, P<Ty>),
    /// A `let pat = expr` expression that is only semantically allowed in the condition
    /// of `if` / `while` expressions. (e.g., `if let 0 = x { .. }`).
    ///
    /// `Span` represents the whole `let pat = expr` statement.
    Let(P<Pat>, P<Expr>, Span, Recovered),
    /// An `if` block, with an optional `else` block.
    ///
    /// `if expr { block } else { expr }`
    If(P<Expr>, P<Block>, Option<P<Expr>>),
    /// A while loop, with an optional label.
    ///
    /// `'label: while expr { block }`
    While(P<Expr>, P<Block>, Option<Label>),
    /// A `for` loop, with an optional label.
    ///
    /// `'label: for await? pat in iter { block }`
    ///
    /// This is desugared to a combination of `loop` and `match` expressions.
    ForLoop { pat: P<Pat>, iter: P<Expr>, body: P<Block>, label: Option<Label>, kind: ForLoopKind },
    /// Conditionless loop (can be exited with `break`, `continue`, or `return`).
    ///
    /// `'label: loop { block }`
    Loop(P<Block>, Option<Label>, Span),
    /// A `match` block.
    Match(P<Expr>, ThinVec<Arm>, MatchKind),
    /// A closure (e.g., `move |a, b, c| a + b + c`).
    Closure(Box<Closure>),
    /// A block (`'label: { ... }`).
    Block(P<Block>, Option<Label>),
    /// An `async` block (`async move { ... }`),
    /// or a `gen` block (`gen move { ... }`).
    ///
    /// The span is the "decl", which is the header before the body `{ }`
    /// including the `asyng`/`gen` keywords and possibly `move`.
    Gen(CaptureBy, P<Block>, GenBlockKind, Span),
    /// An await expression (`my_future.await`). Span is of await keyword.
    Await(P<Expr>, Span),

    /// A try block (`try { ... }`).
    TryBlock(P<Block>),

    /// An assignment (`a = foo()`).
    /// The `Span` argument is the span of the `=` token.
    Assign(P<Expr>, P<Expr>, Span),
    /// An assignment with an operator.
    ///
    /// E.g., `a += 1`.
    AssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named (e.g., `obj.foo`) or unnamed (e.g., `obj.0`) struct field.
    Field(P<Expr>, Ident),
    /// An indexing operation (e.g., `foo[2]`).
    /// The span represents the span of the `[2]`, including brackets.
    Index(P<Expr>, P<Expr>, Span),
    /// A range (e.g., `1..2`, `1..`, `..2`, `1..=2`, `..=2`; and `..` in destructuring assignment).
    Range(Option<P<Expr>>, Option<P<Expr>>, RangeLimits),
    /// An underscore, used in destructuring assignment to ignore a value.
    Underscore,

    /// Variable reference, possibly containing `::` and/or type
    /// parameters (e.g., `foo::bar::<baz>`).
    ///
    /// Optionally "qualified" (e.g., `<Vec<T> as SomeTrait>::SomeType`).
    Path(Option<P<QSelf>>, Path),

    /// A referencing operation (`&a`, `&mut a`, `&raw const a` or `&raw mut a`).
    AddrOf(BorrowKind, Mutability, P<Expr>),
    /// A `break`, with an optional label to break, and an optional expression.
    Break(Option<Label>, Option<P<Expr>>),
    /// A `continue`, with an optional label.
    Continue(Option<Label>),
    /// A `return`, with an optional value to be returned.
    Ret(Option<P<Expr>>),

    /// Output of the `asm!()` macro.
    InlineAsm(P<InlineAsm>),

    /// An `offset_of` expression (e.g., `builtin # offset_of(Struct, field)`).
    ///
    /// Usually not written directly in user code but
    /// indirectly via the macro `core::mem::offset_of!(...)`.
    OffsetOf(P<Ty>, P<[Ident]>),

    /// A macro invocation; pre-expansion.
    MacCall(P<MacCall>),

    /// A struct literal expression.
    ///
    /// E.g., `Foo {x: 1, y: 2}`, or `Foo {x: 1, .. rest}`.
    Struct(P<StructExpr>),

    /// An array literal constructed from one repeated element.
    ///
    /// E.g., `[1; 5]`. The expression is the element to be
    /// repeated; the constant is the number of times to repeat it.
    Repeat(P<Expr>, AnonConst),

    /// No-op: used solely so we can pretty-print faithfully.
    Paren(P<Expr>),

    /// A try expression (`expr?`).
    Try(P<Expr>),

    /// A `yield`, with an optional value to be yielded.
    Yield(Option<P<Expr>>),

    /// A `do yeet` (aka `throw`/`fail`/`bail`/`raise`/whatever),
    /// with an optional value to be returned.
    Yeet(Option<P<Expr>>),

    /// A tail call return, with the value to be returned.
    ///
    /// While `.0` must be a function call, we check this later, after parsing.
    Become(P<Expr>),

    /// Bytes included via `include_bytes!`
    /// Added for optimization purposes to avoid the need to escape
    /// large binary blobs - should always behave like [`ExprKind::Lit`]
    /// with a `ByteStr` literal.
    IncludedBytes(Lrc<[u8]>),

    /// A `format_args!()` expression.
    FormatArgs(P<FormatArgs>),

    /// Placeholder for an expression that wasn't syntactically well formed in some way.
    Err(ErrorGuaranteed),

    /// Acts as a null expression. Lowering it will always emit a bug.
    Dummy,
}

/// Used to differentiate between `for` loops and `for await` loops.
#[derive(Clone, Copy, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum ForLoopKind {
    For,
    ForAwait,
}

/// Used to differentiate between `async {}` blocks and `gen {}` blocks.
#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum GenBlockKind {
    Async,
    Gen,
    AsyncGen,
}

impl fmt::Display for GenBlockKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.modifier().fmt(f)
    }
}

impl GenBlockKind {
    pub fn modifier(&self) -> &'static str {
        match self {
            GenBlockKind::Async => "async",
            GenBlockKind::Gen => "gen",
            GenBlockKind::AsyncGen => "async gen",
        }
    }
}

/// The explicit `Self` type in a "qualified path". The actual
/// path, including the trait and the associated item, is stored
/// separately. `position` represents the index of the associated
/// item qualified with this `Self` type.
///
/// ```ignore (only-for-syntax-highlight)
/// <Vec<T> as a::b::Trait>::AssociatedItem
///  ^~~~~     ~~~~~~~~~~~~~~^
///  ty        position = 3
///
/// <Vec<T>>::AssociatedItem
///  ^~~~~    ^
///  ty       position = 0
/// ```
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct QSelf {
    pub ty: P<Ty>,

    /// The span of `a::b::Trait` in a path like `<Vec<T> as
    /// a::b::Trait>::AssociatedItem`; in the case where `position ==
    /// 0`, this is an empty span.
    pub path_span: Span,
    pub position: usize,
}

/// A capture clause used in closures and `async` blocks.
#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum CaptureBy {
    /// `move |x| y + x`.
    Value {
        /// The span of the `move` keyword.
        move_kw: Span,
    },
    /// `move` keyword was not specified.
    Ref,
}

/// Closure lifetime binder, `for<'a, 'b>` in `for<'a, 'b> |_: &'a (), _: &'b ()|`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ClosureBinder {
    /// The binder is not present, all closure lifetimes are inferred.
    NotPresent,
    /// The binder is present.
    For {
        /// Span of the whole `for<>` clause
        ///
        /// ```text
        /// for<'a, 'b> |_: &'a (), _: &'b ()| { ... }
        /// ^^^^^^^^^^^ -- this
        /// ```
        span: Span,

        /// Lifetimes in the `for<>` closure
        ///
        /// ```text
        /// for<'a, 'b> |_: &'a (), _: &'b ()| { ... }
        ///     ^^^^^^ -- this
        /// ```
        generic_params: ThinVec<GenericParam>,
    },
}

/// Represents a macro invocation. The `path` indicates which macro
/// is being invoked, and the `args` are arguments passed to it.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MacCall {
    pub path: Path,
    pub args: P<DelimArgs>,
}

impl MacCall {
    pub fn span(&self) -> Span {
        self.path.span.to(self.args.dspan.entire())
    }
}

/// Arguments passed to an attribute macro.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrArgs {
    /// No arguments: `#[attr]`.
    Empty,
    /// Delimited arguments: `#[attr()/[]/{}]`.
    Delimited(DelimArgs),
    /// Arguments of a key-value attribute: `#[attr = "value"]`.
    Eq(
        /// Span of the `=` token.
        Span,
        /// The "value".
        AttrArgsEq,
    ),
}

// The RHS of an `AttrArgs::Eq` starts out as an expression. Once macro
// expansion is completed, all cases end up either as a meta item literal,
// which is the form used after lowering to HIR, or as an error.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrArgsEq {
    Ast(P<Expr>),
    Hir(MetaItemLit),
}

impl AttrArgs {
    pub fn span(&self) -> Option<Span> {
        match self {
            AttrArgs::Empty => None,
            AttrArgs::Delimited(args) => Some(args.dspan.entire()),
            AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => Some(eq_span.to(expr.span)),
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when getting span: {:?}", lit);
            }
        }
    }

    /// Tokens inside the delimiters or after `=`.
    /// Proc macros see these tokens, for example.
    pub fn inner_tokens(&self) -> TokenStream {
        match self {
            AttrArgs::Empty => TokenStream::default(),
            AttrArgs::Delimited(args) => args.tokens.clone(),
            AttrArgs::Eq(_, AttrArgsEq::Ast(expr)) => TokenStream::from_ast(expr),
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when getting inner tokens: {:?}", lit)
            }
        }
    }
}

impl<CTX> HashStable<CTX> for AttrArgs
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(ctx, hasher);
        match self {
            AttrArgs::Empty => {}
            AttrArgs::Delimited(args) => args.hash_stable(ctx, hasher),
            AttrArgs::Eq(_eq_span, AttrArgsEq::Ast(expr)) => {
                unreachable!("hash_stable {:?}", expr);
            }
            AttrArgs::Eq(eq_span, AttrArgsEq::Hir(lit)) => {
                eq_span.hash_stable(ctx, hasher);
                lit.hash_stable(ctx, hasher);
            }
        }
    }
}

/// Delimited arguments, as used in `#[attr()/[]/{}]` or `mac!()/[]/{}`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct DelimArgs {
    pub dspan: DelimSpan,
    pub delim: Delimiter, // Note: `Delimiter::Invisible` never occurs
    pub tokens: TokenStream,
}

impl DelimArgs {
    /// Whether a macro with these arguments needs a semicolon
    /// when used as a standalone item or statement.
    pub fn need_semicolon(&self) -> bool {
        !matches!(self, DelimArgs { delim: Delimiter::Brace, .. })
    }
}

impl<CTX> HashStable<CTX> for DelimArgs
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let DelimArgs { dspan, delim, tokens } = self;
        dspan.hash_stable(ctx, hasher);
        delim.hash_stable(ctx, hasher);
        tokens.hash_stable(ctx, hasher);
    }
}

/// Represents a macro definition.
#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MacroDef {
    pub body: P<DelimArgs>,
    /// `true` if macro was defined with `macro_rules`.
    pub macro_rules: bool,
}

#[derive(Clone, Encodable, Decodable, Debug, Copy, Hash, Eq, PartialEq)]
#[derive(HashStable_Generic)]
pub enum StrStyle {
    /// A regular string, like `"foo"`.
    Cooked,
    /// A raw string, like `r##"foo"##`.
    ///
    /// The value is the number of `#` symbols used.
    Raw(u8),
}

/// The kind of match expression
#[derive(Clone, Copy, Encodable, Decodable, Debug, PartialEq)]
pub enum MatchKind {
    /// match expr { ... }
    Prefix,
    /// expr.match { ... }
    Postfix,
}

/// A literal in a meta item.
#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MetaItemLit {
    /// The original literal as written in the source code.
    pub symbol: Symbol,
    /// The original suffix as written in the source code.
    pub suffix: Option<Symbol>,
    /// The "semantic" representation of the literal lowered from the original tokens.
    /// Strings are unescaped, hexadecimal forms are eliminated, etc.
    pub kind: LitKind,
    pub span: Span,
}

/// Similar to `MetaItemLit`, but restricted to string literals.
#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub struct StrLit {
    /// The original literal as written in source code.
    pub symbol: Symbol,
    /// The original suffix as written in source code.
    pub suffix: Option<Symbol>,
    /// The semantic (unescaped) representation of the literal.
    pub symbol_unescaped: Symbol,
    pub style: StrStyle,
    pub span: Span,
}

impl StrLit {
    pub fn as_token_lit(&self) -> token::Lit {
        let token_kind = match self.style {
            StrStyle::Cooked => token::Str,
            StrStyle::Raw(n) => token::StrRaw(n),
        };
        token::Lit::new(token_kind, self.symbol, self.suffix)
    }
}

/// Type of the integer literal based on provided suffix.
#[derive(Clone, Copy, Encodable, Decodable, Debug, Hash, Eq, PartialEq)]
#[derive(HashStable_Generic)]
pub enum LitIntType {
    /// e.g. `42_i32`.
    Signed(IntTy),
    /// e.g. `42_u32`.
    Unsigned(UintTy),
    /// e.g. `42`.
    Unsuffixed,
}

/// Type of the float literal based on provided suffix.
#[derive(Clone, Copy, Encodable, Decodable, Debug, Hash, Eq, PartialEq)]
#[derive(HashStable_Generic)]
pub enum LitFloatType {
    /// A float literal with a suffix (`1f32` or `1E10f32`).
    Suffixed(FloatTy),
    /// A float literal without a suffix (`1.0 or 1.0E10`).
    Unsuffixed,
}

/// This type is used within both `ast::MetaItemLit` and `hir::Lit`.
///
/// Note that the entire literal (including the suffix) is considered when
/// deciding the `LitKind`. This means that float literals like `1f32` are
/// classified by this type as `Float`. This is different to `token::LitKind`
/// which does *not* consider the suffix.
#[derive(Clone, Encodable, Decodable, Debug, Hash, Eq, PartialEq, HashStable_Generic)]
pub enum LitKind {
    /// A string literal (`"foo"`). The symbol is unescaped, and so may differ
    /// from the original token's symbol.
    Str(Symbol, StrStyle),
    /// A byte string (`b"foo"`). Not stored as a symbol because it might be
    /// non-utf8, and symbols only allow utf8 strings.
    ByteStr(Lrc<[u8]>, StrStyle),
    /// A C String (`c"foo"`). Guaranteed to only have `\0` at the end.
    CStr(Lrc<[u8]>, StrStyle),
    /// A byte char (`b'f'`).
    Byte(u8),
    /// A character literal (`'a'`).
    Char(char),
    /// An integer literal (`1`).
    Int(Pu128, LitIntType),
    /// A float literal (`1.0`, `1f64` or `1E10f64`). The pre-suffix part is
    /// stored as a symbol rather than `f64` so that `LitKind` can impl `Eq`
    /// and `Hash`.
    Float(Symbol, LitFloatType),
    /// A boolean literal (`true`, `false`).
    Bool(bool),
    /// Placeholder for a literal that wasn't well-formed in some way.
    Err(ErrorGuaranteed),
}

impl LitKind {
    pub fn str(&self) -> Option<Symbol> {
        match *self {
            LitKind::Str(s, _) => Some(s),
            _ => None,
        }
    }

    /// Returns `true` if this literal is a string.
    pub fn is_str(&self) -> bool {
        matches!(self, LitKind::Str(..))
    }

    /// Returns `true` if this literal is byte literal string.
    pub fn is_bytestr(&self) -> bool {
        matches!(self, LitKind::ByteStr(..))
    }

    /// Returns `true` if this is a numeric literal.
    pub fn is_numeric(&self) -> bool {
        matches!(self, LitKind::Int(..) | LitKind::Float(..))
    }

    /// Returns `true` if this literal has no suffix.
    /// Note: this will return true for literals with prefixes such as raw strings and byte strings.
    pub fn is_unsuffixed(&self) -> bool {
        !self.is_suffixed()
    }

    /// Returns `true` if this literal has a suffix.
    pub fn is_suffixed(&self) -> bool {
        match *self {
            // suffixed variants
            LitKind::Int(_, LitIntType::Signed(..) | LitIntType::Unsigned(..))
            | LitKind::Float(_, LitFloatType::Suffixed(..)) => true,
            // unsuffixed variants
            LitKind::Str(..)
            | LitKind::ByteStr(..)
            | LitKind::CStr(..)
            | LitKind::Byte(..)
            | LitKind::Char(..)
            | LitKind::Int(_, LitIntType::Unsuffixed)
            | LitKind::Float(_, LitFloatType::Unsuffixed)
            | LitKind::Bool(..)
            | LitKind::Err(_) => false,
        }
    }
}

// N.B., If you change this, you'll probably want to change the corresponding
// type structure in `middle/ty.rs` as well.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

/// Represents a function's signature in a trait declaration,
/// trait implementation, or free function.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FnSig {
    pub header: FnHeader,
    pub decl: P<FnDecl>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}

impl FloatTy {
    pub fn name_str(self) -> &'static str {
        match self {
            FloatTy::F16 => "f16",
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
            FloatTy::F128 => "f128",
        }
    }

    pub fn name(self) -> Symbol {
        match self {
            FloatTy::F16 => sym::f16,
            FloatTy::F32 => sym::f32,
            FloatTy::F64 => sym::f64,
            FloatTy::F128 => sym::f128,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn name(&self) -> Symbol {
        match *self {
            IntTy::Isize => sym::isize,
            IntTy::I8 => sym::i8,
            IntTy::I16 => sym::i16,
            IntTy::I32 => sym::i32,
            IntTy::I64 => sym::i64,
            IntTy::I128 => sym::i128,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn name(&self) -> Symbol {
        match *self {
            UintTy::Usize => sym::usize,
            UintTy::U8 => sym::u8,
            UintTy::U16 => sym::u16,
            UintTy::U32 => sym::u32,
            UintTy::U64 => sym::u64,
            UintTy::U128 => sym::u128,
        }
    }
}

/// A constraint on an associated item.
///
/// ### Examples
///
/// * the `A = Ty` and `B = Ty` in `Trait<A = Ty, B = Ty>`
/// * the `G<Ty> = Ty` in `Trait<G<Ty> = Ty>`
/// * the `A: Bound` in `Trait<A: Bound>`
/// * the `RetTy` in `Trait(ArgTy, ArgTy) -> RetTy`
/// * the `C = { Ct }` in `Trait<C = { Ct }>` (feature `associated_const_equality`)
/// * the `f(..): Bound` in `Trait<f(..): Bound>` (feature `return_type_notation`)
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct AssocItemConstraint {
    pub id: NodeId,
    pub ident: Ident,
    pub gen_args: Option<GenericArgs>,
    pub kind: AssocItemConstraintKind,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum Term {
    Ty(P<Ty>),
    Const(AnonConst),
}

impl From<P<Ty>> for Term {
    fn from(v: P<Ty>) -> Self {
        Term::Ty(v)
    }
}

impl From<AnonConst> for Term {
    fn from(v: AnonConst) -> Self {
        Term::Const(v)
    }
}

/// The kind of [associated item constraint][AssocItemConstraint].
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AssocItemConstraintKind {
    /// An equality constraint for an associated item (e.g., `AssocTy = Ty` in `Trait<AssocTy = Ty>`).
    ///
    /// Also known as an *associated item binding* (we *bind* an associated item to a term).
    ///
    /// Furthermore, associated type equality constraints can also be referred to as *associated type
    /// bindings*. Similarly with associated const equality constraints and *associated const bindings*.
    Equality { term: Term },
    /// A bound on an associated type (e.g., `AssocTy: Bound` in `Trait<AssocTy: Bound>`).
    Bound { bounds: GenericBounds },
}

#[derive(Encodable, Decodable, Debug)]
pub struct Ty {
    pub id: NodeId,
    pub kind: TyKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Clone for Ty {
    fn clone(&self) -> Self {
        ensure_sufficient_stack(|| Self {
            id: self.id,
            kind: self.kind.clone(),
            span: self.span,
            tokens: self.tokens.clone(),
        })
    }
}

impl Ty {
    pub fn peel_refs(&self) -> &Self {
        let mut final_ty = self;
        while let TyKind::Ref(_, MutTy { ty, .. }) | TyKind::Ptr(MutTy { ty, .. }) = &final_ty.kind
        {
            final_ty = ty;
        }
        final_ty
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct BareFnTy {
    pub safety: Safety,
    pub ext: Extern,
    pub generic_params: ThinVec<GenericParam>,
    pub decl: P<FnDecl>,
    /// Span of the `[unsafe] [extern] fn(...) -> ...` part, i.e. everything
    /// after the generic params (if there are any, e.g. `for<'a>`).
    pub decl_span: Span,
}

/// The various kinds of type recognized by the compiler.
//
// Adding a new variant? Please update `test_ty` in `tests/ui/macros/stringify.rs`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum TyKind {
    /// A variable-length slice (`[T]`).
    Slice(P<Ty>),
    /// A fixed length array (`[T; n]`).
    Array(P<Ty>, AnonConst),
    /// A raw pointer (`*const T` or `*mut T`).
    Ptr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`).
    Ref(Option<Lifetime>, MutTy),
    /// A bare function (e.g., `fn(usize) -> bool`).
    BareFn(P<BareFnTy>),
    /// The never type (`!`).
    Never,
    /// A tuple (`(A, B, C, D,...)`).
    Tup(ThinVec<P<Ty>>),
    /// An anonymous struct type i.e. `struct { foo: Type }`.
    AnonStruct(NodeId, ThinVec<FieldDef>),
    /// An anonymous union type i.e. `union { bar: Type }`.
    AnonUnion(NodeId, ThinVec<FieldDef>),
    /// A path (`module::module::...::Type`), optionally
    /// "qualified", e.g., `<Vec<T> as SomeTrait>::SomeType`.
    ///
    /// Type parameters are stored in the `Path` itself.
    Path(Option<P<QSelf>>, Path),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(GenericBounds, TraitObjectSyntax),
    /// An `impl Bound1 + Bound2 + Bound3` type
    /// where `Bound` is a trait or a lifetime.
    ///
    /// The `NodeId` exists to prevent lowering from having to
    /// generate `NodeId`s on the fly, which would complicate
    /// the generation of opaque `type Foo = impl Trait` items significantly.
    ImplTrait(NodeId, GenericBounds),
    /// No-op; kept solely so that we can pretty-print faithfully.
    Paren(P<Ty>),
    /// Unused for now.
    Typeof(AnonConst),
    /// This means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Inferred type of a `self` or `&self` argument in a method.
    ImplicitSelf,
    /// A macro in the type position.
    MacCall(P<MacCall>),
    /// Placeholder for a `va_list`.
    CVarArgs,
    /// Pattern types like `pattern_type!(u32 is 1..=)`, which is the same as `NonZero<u32>`,
    /// just as part of the type system.
    Pat(P<Ty>, P<Pat>),
    /// Sometimes we need a dummy value when no error has occurred.
    Dummy,
    /// Placeholder for a kind that has failed to be defined.
    Err(ErrorGuaranteed),
}

impl TyKind {
    pub fn is_implicit_self(&self) -> bool {
        matches!(self, TyKind::ImplicitSelf)
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, TyKind::Tup(tys) if tys.is_empty())
    }

    pub fn is_simple_path(&self) -> Option<Symbol> {
        if let TyKind::Path(None, Path { segments, .. }) = &self
            && let [segment] = &segments[..]
            && segment.args.is_none()
        {
            Some(segment.ident.name)
        } else {
            None
        }
    }

    pub fn is_anon_adt(&self) -> bool {
        matches!(self, TyKind::AnonStruct(..) | TyKind::AnonUnion(..))
    }
}

/// Syntax used to declare a trait object.
#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum TraitObjectSyntax {
    Dyn,
    DynStar,
    None,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum PreciseCapturingArg {
    /// Lifetime parameter.
    Lifetime(Lifetime),
    /// Type or const parameter.
    Arg(Path, NodeId),
}

/// Inline assembly operand explicit register or register class.
///
/// E.g., `"eax"` as in `asm!("mov eax, 2", out("eax") result)`.
#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub enum InlineAsmRegOrRegClass {
    Reg(Symbol),
    RegClass(Symbol),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Encodable, Decodable, HashStable_Generic)]
pub struct InlineAsmOptions(u16);
bitflags::bitflags! {
    impl InlineAsmOptions: u16 {
        const PURE            = 1 << 0;
        const NOMEM           = 1 << 1;
        const READONLY        = 1 << 2;
        const PRESERVES_FLAGS = 1 << 3;
        const NORETURN        = 1 << 4;
        const NOSTACK         = 1 << 5;
        const ATT_SYNTAX      = 1 << 6;
        const RAW             = 1 << 7;
        const MAY_UNWIND      = 1 << 8;
    }
}

impl InlineAsmOptions {
    pub fn human_readable_names(&self) -> Vec<&'static str> {
        let mut options = vec![];

        if self.contains(InlineAsmOptions::PURE) {
            options.push("pure");
        }
        if self.contains(InlineAsmOptions::NOMEM) {
            options.push("nomem");
        }
        if self.contains(InlineAsmOptions::READONLY) {
            options.push("readonly");
        }
        if self.contains(InlineAsmOptions::PRESERVES_FLAGS) {
            options.push("preserves_flags");
        }
        if self.contains(InlineAsmOptions::NORETURN) {
            options.push("noreturn");
        }
        if self.contains(InlineAsmOptions::NOSTACK) {
            options.push("nostack");
        }
        if self.contains(InlineAsmOptions::ATT_SYNTAX) {
            options.push("att_syntax");
        }
        if self.contains(InlineAsmOptions::RAW) {
            options.push("raw");
        }
        if self.contains(InlineAsmOptions::MAY_UNWIND) {
            options.push("may_unwind");
        }

        options
    }
}

impl std::fmt::Debug for InlineAsmOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Hash, HashStable_Generic)]
pub enum InlineAsmTemplatePiece {
    String(Cow<'static, str>),
    Placeholder { operand_idx: usize, modifier: Option<char>, span: Span },
}

impl fmt::Display for InlineAsmTemplatePiece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => {
                for c in s.chars() {
                    match c {
                        '{' => f.write_str("{{")?,
                        '}' => f.write_str("}}")?,
                        _ => c.fmt(f)?,
                    }
                }
                Ok(())
            }
            Self::Placeholder { operand_idx, modifier: Some(modifier), .. } => {
                write!(f, "{{{operand_idx}:{modifier}}}")
            }
            Self::Placeholder { operand_idx, modifier: None, .. } => {
                write!(f, "{{{operand_idx}}}")
            }
        }
    }
}

impl InlineAsmTemplatePiece {
    /// Rebuilds the asm template string from its pieces.
    pub fn to_string(s: &[Self]) -> String {
        use fmt::Write;
        let mut out = String::new();
        for p in s.iter() {
            let _ = write!(out, "{p}");
        }
        out
    }
}

/// Inline assembly symbol operands get their own AST node that is somewhat
/// similar to `AnonConst`.
///
/// The main difference is that we specifically don't assign it `DefId` in
/// `DefCollector`. Instead this is deferred until AST lowering where we
/// lower it to an `AnonConst` (for functions) or a `Path` (for statics)
/// depending on what the path resolves to.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct InlineAsmSym {
    pub id: NodeId,
    pub qself: Option<P<QSelf>>,
    pub path: Path,
}

/// Inline assembly operand.
///
/// E.g., `out("eax") result` as in `asm!("mov eax, 2", out("eax") result)`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum InlineAsmOperand {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: P<Expr>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<P<Expr>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: P<Expr>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: P<Expr>,
        out_expr: Option<P<Expr>>,
    },
    Const {
        anon_const: AnonConst,
    },
    Sym {
        sym: InlineAsmSym,
    },
    Label {
        block: P<Block>,
    },
}

impl InlineAsmOperand {
    pub fn reg(&self) -> Option<&InlineAsmRegOrRegClass> {
        match self {
            Self::In { reg, .. }
            | Self::Out { reg, .. }
            | Self::InOut { reg, .. }
            | Self::SplitInOut { reg, .. } => Some(reg),
            Self::Const { .. } | Self::Sym { .. } | Self::Label { .. } => None,
        }
    }
}

/// Inline assembly.
///
/// E.g., `asm!("NOP");`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct InlineAsm {
    pub template: Vec<InlineAsmTemplatePiece>,
    pub template_strs: Box<[(Symbol, Option<Symbol>, Span)]>,
    pub operands: Vec<(InlineAsmOperand, Span)>,
    pub clobber_abis: Vec<(Symbol, Span)>,
    pub options: InlineAsmOptions,
    pub line_spans: Vec<Span>,
}

/// A parameter in a function header.
///
/// E.g., `bar: usize` as in `fn foo(bar: usize)`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Param {
    pub attrs: AttrVec,
    pub ty: P<Ty>,
    pub pat: P<Pat>,
    pub id: NodeId,
    pub span: Span,
    pub is_placeholder: bool,
}

/// Alternative representation for `Arg`s describing `self` parameter of methods.
///
/// E.g., `&mut self` as in `fn foo(&mut self)`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum SelfKind {
    /// `self`, `mut self`
    Value(Mutability),
    /// `&'lt self`, `&'lt mut self`
    Region(Option<Lifetime>, Mutability),
    /// `self: TYPE`, `mut self: TYPE`
    Explicit(P<Ty>, Mutability),
}

pub type ExplicitSelf = Spanned<SelfKind>;

impl Param {
    /// Attempts to cast parameter to `ExplicitSelf`.
    pub fn to_self(&self) -> Option<ExplicitSelf> {
        if let PatKind::Ident(BindingMode(ByRef::No, mutbl), ident, _) = self.pat.kind {
            if ident.name == kw::SelfLower {
                return match self.ty.kind {
                    TyKind::ImplicitSelf => Some(respan(self.pat.span, SelfKind::Value(mutbl))),
                    TyKind::Ref(lt, MutTy { ref ty, mutbl }) if ty.kind.is_implicit_self() => {
                        Some(respan(self.pat.span, SelfKind::Region(lt, mutbl)))
                    }
                    _ => Some(respan(
                        self.pat.span.to(self.ty.span),
                        SelfKind::Explicit(self.ty.clone(), mutbl),
                    )),
                };
            }
        }
        None
    }

    /// Returns `true` if parameter is `self`.
    pub fn is_self(&self) -> bool {
        if let PatKind::Ident(_, ident, _) = self.pat.kind {
            ident.name == kw::SelfLower
        } else {
            false
        }
    }

    /// Builds a `Param` object from `ExplicitSelf`.
    pub fn from_self(attrs: AttrVec, eself: ExplicitSelf, eself_ident: Ident) -> Param {
        let span = eself.span.to(eself_ident.span);
        let infer_ty = P(Ty {
            id: DUMMY_NODE_ID,
            kind: TyKind::ImplicitSelf,
            span: eself_ident.span,
            tokens: None,
        });
        let (mutbl, ty) = match eself.node {
            SelfKind::Explicit(ty, mutbl) => (mutbl, ty),
            SelfKind::Value(mutbl) => (mutbl, infer_ty),
            SelfKind::Region(lt, mutbl) => (
                Mutability::Not,
                P(Ty {
                    id: DUMMY_NODE_ID,
                    kind: TyKind::Ref(lt, MutTy { ty: infer_ty, mutbl }),
                    span,
                    tokens: None,
                }),
            ),
        };
        Param {
            attrs,
            pat: P(Pat {
                id: DUMMY_NODE_ID,
                kind: PatKind::Ident(BindingMode(ByRef::No, mutbl), eself_ident, None),
                span,
                tokens: None,
            }),
            span,
            ty,
            id: DUMMY_NODE_ID,
            is_placeholder: false,
        }
    }
}

/// A signature (not the body) of a function declaration.
///
/// E.g., `fn foo(bar: baz)`.
///
/// Please note that it's different from `FnHeader` structure
/// which contains metadata about function safety, asyncness, constness and ABI.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FnDecl {
    pub inputs: ThinVec<Param>,
    pub output: FnRetTy,
}

impl FnDecl {
    pub fn has_self(&self) -> bool {
        self.inputs.get(0).is_some_and(Param::is_self)
    }
    pub fn c_variadic(&self) -> bool {
        self.inputs.last().is_some_and(|arg| matches!(arg.ty.kind, TyKind::CVarArgs))
    }
}

/// Is the trait definition an auto trait?
#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum IsAuto {
    Yes,
    No,
}

/// Safety of items.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum Safety {
    /// `unsafe` an item is explicitly marked as `unsafe`.
    Unsafe(Span),
    /// `safe` an item is explicitly marked as `safe`.
    Safe(Span),
    /// Default means no value was provided, it will take a default value given the context in
    /// which is used.
    Default,
}

/// Describes what kind of coroutine markers, if any, a function has.
///
/// Coroutine markers are things that cause the function to generate a coroutine, such as `async`,
/// which makes the function return `impl Future`, or `gen`, which makes the function return `impl
/// Iterator`.
#[derive(Copy, Clone, Encodable, Decodable, Debug)]
pub enum CoroutineKind {
    /// `async`, which returns an `impl Future`.
    Async { span: Span, closure_id: NodeId, return_impl_trait_id: NodeId },
    /// `gen`, which returns an `impl Iterator`.
    Gen { span: Span, closure_id: NodeId, return_impl_trait_id: NodeId },
    /// `async gen`, which returns an `impl AsyncIterator`.
    AsyncGen { span: Span, closure_id: NodeId, return_impl_trait_id: NodeId },
}

impl CoroutineKind {
    pub fn span(self) -> Span {
        match self {
            CoroutineKind::Async { span, .. } => span,
            CoroutineKind::Gen { span, .. } => span,
            CoroutineKind::AsyncGen { span, .. } => span,
        }
    }

    pub fn is_async(self) -> bool {
        matches!(self, CoroutineKind::Async { .. })
    }

    pub fn is_gen(self) -> bool {
        matches!(self, CoroutineKind::Gen { .. })
    }

    pub fn closure_id(self) -> NodeId {
        match self {
            CoroutineKind::Async { closure_id, .. }
            | CoroutineKind::Gen { closure_id, .. }
            | CoroutineKind::AsyncGen { closure_id, .. } => closure_id,
        }
    }

    /// In this case this is an `async` or `gen` return, the `NodeId` for the generated `impl Trait`
    /// item.
    pub fn return_id(self) -> (NodeId, Span) {
        match self {
            CoroutineKind::Async { return_impl_trait_id, span, .. }
            | CoroutineKind::Gen { return_impl_trait_id, span, .. }
            | CoroutineKind::AsyncGen { return_impl_trait_id, span, .. } => {
                (return_impl_trait_id, span)
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum Const {
    Yes(Span),
    No,
}

/// Item defaultness.
/// For details see the [RFC #2532](https://github.com/rust-lang/rfcs/pull/2532).
#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Defaultness {
    Default(Span),
    Final,
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative(Span),
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ImplPolarity::Positive => "positive".fmt(f),
            ImplPolarity::Negative(_) => "negative".fmt(f),
        }
    }
}

/// The polarity of a trait bound.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum BoundPolarity {
    /// `Type: Trait`
    Positive,
    /// `Type: !Trait`
    Negative(Span),
    /// `Type: ?Trait`
    Maybe(Span),
}

impl BoundPolarity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "",
            Self::Negative(_) => "!",
            Self::Maybe(_) => "?",
        }
    }
}

/// The constness of a trait bound.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum BoundConstness {
    /// `Type: Trait`
    Never,
    /// `Type: const Trait`
    Always(Span),
    /// `Type: ~const Trait`
    Maybe(Span),
}

impl BoundConstness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Never => "",
            Self::Always(_) => "const",
            Self::Maybe(_) => "~const",
        }
    }
}

/// The asyncness of a trait bound.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum BoundAsyncness {
    /// `Type: Trait`
    Normal,
    /// `Type: async Trait`
    Async(Span),
}

impl BoundAsyncness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "",
            Self::Async(_) => "async",
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FnRetTy {
    /// Returns type is not specified.
    ///
    /// Functions default to `()` and closures default to inference.
    /// Span points to where return type would be inserted.
    Default(Span),
    /// Everything else.
    Ty(P<Ty>),
}

impl FnRetTy {
    pub fn span(&self) -> Span {
        match self {
            &FnRetTy::Default(span) => span,
            FnRetTy::Ty(ty) => ty.span,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug)]
pub enum Inline {
    Yes,
    No,
}

/// Module item kind.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ModKind {
    /// Module with inlined definition `mod foo { ... }`,
    /// or with definition outlined to a separate file `mod foo;` and already loaded from it.
    /// The inner span is from the first token past `{` to the last token until `}`,
    /// or from the first to the last token in the loaded file.
    Loaded(ThinVec<P<Item>>, Inline, ModSpans),
    /// Module with definition outlined to a separate file `mod foo;` but not yet loaded from it.
    Unloaded,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, Default)]
pub struct ModSpans {
    /// `inner_span` covers the body of the module; for a file module, its the whole file.
    /// For an inline module, its the span inside the `{ ... }`, not including the curly braces.
    pub inner_span: Span,
    pub inject_use_span: Span,
}

/// Foreign module declaration.
///
/// E.g., `extern { .. }` or `extern "C" { .. }`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ForeignMod {
    /// `unsafe` keyword accepted syntactically for macro DSLs, but not
    /// semantically by Rust.
    pub safety: Safety,
    pub abi: Option<StrLit>,
    pub items: ThinVec<P<ForeignItem>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct EnumDef {
    pub variants: ThinVec<Variant>,
}
/// Enum variant.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Variant {
    /// Attributes of the variant.
    pub attrs: AttrVec,
    /// Id of the variant (not the constructor, see `VariantData::ctor_id()`).
    pub id: NodeId,
    /// Span
    pub span: Span,
    /// The visibility of the variant. Syntactically accepted but not semantically.
    pub vis: Visibility,
    /// Name of the variant.
    pub ident: Ident,

    /// Fields and constructor id of the variant.
    pub data: VariantData,
    /// Explicit discriminant, e.g., `Foo = 1`.
    pub disr_expr: Option<AnonConst>,
    /// Is a macro placeholder.
    pub is_placeholder: bool,
}

/// Part of `use` item to the right of its prefix.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum UseTreeKind {
    /// `use prefix` or `use prefix as rename`
    Simple(Option<Ident>),
    /// `use prefix::{...}`
    ///
    /// The span represents the braces of the nested group and all elements within:
    ///
    /// ```text
    /// use foo::{bar, baz};
    ///          ^^^^^^^^^^
    /// ```
    Nested { items: ThinVec<(UseTree, NodeId)>, span: Span },
    /// `use prefix::*`
    Glob,
}

/// A tree of paths sharing common prefixes.
/// Used in `use` items both at top-level and inside of braces in import groups.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct UseTree {
    pub prefix: Path,
    pub kind: UseTreeKind,
    pub span: Span,
}

impl UseTree {
    pub fn ident(&self) -> Ident {
        match self.kind {
            UseTreeKind::Simple(Some(rename)) => rename,
            UseTreeKind::Simple(None) => {
                self.prefix.segments.last().expect("empty prefix in a simple import").ident
            }
            _ => panic!("`UseTree::ident` can only be used on a simple import"),
        }
    }
}

/// Distinguishes between `Attribute`s that decorate items and Attributes that
/// are contained as statements within items. These two cases need to be
/// distinguished for pretty-printing.
#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy, HashStable_Generic)]
pub enum AttrStyle {
    Outer,
    Inner,
}

/// A list of attributes.
pub type AttrVec = ThinVec<Attribute>;

/// A syntax-level representation of an attribute.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Attribute {
    pub kind: AttrKind,
    pub id: AttrId,
    /// Denotes if the attribute decorates the following construct (outer)
    /// or the construct this attribute is contained within (inner).
    pub style: AttrStyle,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrKind {
    /// A normal attribute.
    Normal(P<NormalAttr>),

    /// A doc comment (e.g. `/// ...`, `//! ...`, `/** ... */`, `/*! ... */`).
    /// Doc attributes (e.g. `#[doc="..."]`) are represented with the `Normal`
    /// variant (which is much less compact and thus more expensive).
    DocComment(CommentKind, Symbol),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct NormalAttr {
    pub item: AttrItem,
    // Tokens for the full attribute, e.g. `#[foo]`, `#![bar]`.
    pub tokens: Option<LazyAttrTokenStream>,
}

impl NormalAttr {
    pub fn from_ident(ident: Ident) -> Self {
        Self {
            item: AttrItem {
                unsafety: Safety::Default,
                path: Path::from_ident(ident),
                args: AttrArgs::Empty,
                tokens: None,
            },
            tokens: None,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AttrItem {
    pub unsafety: Safety,
    pub path: Path,
    pub args: AttrArgs,
    // Tokens for the meta item, e.g. just the `foo` within `#[foo]` or `#![foo]`.
    pub tokens: Option<LazyAttrTokenStream>,
}

/// `TraitRef`s appear in impls.
///
/// Resolution maps each `TraitRef`'s `ref_id` to its defining trait; that's all
/// that the `ref_id` is for. The `impl_id` maps to the "self type" of this impl.
/// If this impl is an `ItemKind::Impl`, the `impl_id` is redundant (it could be the
/// same as the impl's `NodeId`).
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PolyTraitRef {
    /// The `'a` in `for<'a> Foo<&'a T>`.
    pub bound_generic_params: ThinVec<GenericParam>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`.
    pub trait_ref: TraitRef,

    pub span: Span,
}

impl PolyTraitRef {
    pub fn new(generic_params: ThinVec<GenericParam>, path: Path, span: Span) -> Self {
        PolyTraitRef {
            bound_generic_params: generic_params,
            trait_ref: TraitRef { path, ref_id: DUMMY_NODE_ID },
            span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Visibility {
    pub kind: VisibilityKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum VisibilityKind {
    Public,
    Restricted { path: P<Path>, id: NodeId, shorthand: bool },
    Inherited,
}

impl VisibilityKind {
    pub fn is_pub(&self) -> bool {
        matches!(self, VisibilityKind::Public)
    }
}

/// Field definition in a struct, variant or union.
///
/// E.g., `bar: usize` as in `struct Foo { bar: usize }`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FieldDef {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
    pub ident: Option<Ident>,

    pub ty: P<Ty>,
    pub is_placeholder: bool,
}

/// Was parsing recovery performed?
#[derive(Copy, Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Recovered {
    No,
    Yes(ErrorGuaranteed),
}

/// Fields and constructor ids of enum variants and structs.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum VariantData {
    /// Struct variant.
    ///
    /// E.g., `Bar { .. }` as in `enum Foo { Bar { .. } }`.
    Struct { fields: ThinVec<FieldDef>, recovered: Recovered },
    /// Tuple variant.
    ///
    /// E.g., `Bar(..)` as in `enum Foo { Bar(..) }`.
    Tuple(ThinVec<FieldDef>, NodeId),
    /// Unit variant.
    ///
    /// E.g., `Bar = ..` as in `enum Foo { Bar = .. }`.
    Unit(NodeId),
}

impl VariantData {
    /// Return the fields of this variant.
    pub fn fields(&self) -> &[FieldDef] {
        match self {
            VariantData::Struct { fields, .. } | VariantData::Tuple(fields, _) => fields,
            _ => &[],
        }
    }

    /// Return the `NodeId` of this variant's constructor, if it has one.
    pub fn ctor_node_id(&self) -> Option<NodeId> {
        match *self {
            VariantData::Struct { .. } => None,
            VariantData::Tuple(_, id) | VariantData::Unit(id) => Some(id),
        }
    }
}

/// An item definition.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Item<K = ItemKind> {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
    /// The name of the item.
    /// It might be a dummy name in case of anonymous items.
    pub ident: Ident,

    pub kind: K,

    /// Original tokens this item was parsed from. This isn't necessarily
    /// available for all items, although over time more and more items should
    /// have this be `Some`. Right now this is primarily used for procedural
    /// macros, notably custom attributes.
    ///
    /// Note that the tokens here do not include the outer attributes, but will
    /// include inner attributes.
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Item {
    /// Return the span that encompasses the attributes.
    pub fn span_with_attributes(&self) -> Span {
        self.attrs.iter().fold(self.span, |acc, attr| acc.to(attr.span))
    }

    pub fn opt_generics(&self) -> Option<&Generics> {
        match &self.kind {
            ItemKind::ExternCrate(_)
            | ItemKind::Use(_)
            | ItemKind::Mod(_, _)
            | ItemKind::ForeignMod(_)
            | ItemKind::GlobalAsm(_)
            | ItemKind::MacCall(_)
            | ItemKind::Delegation(_)
            | ItemKind::DelegationMac(_)
            | ItemKind::MacroDef(_) => None,
            ItemKind::Static(_) => None,
            ItemKind::Const(i) => Some(&i.generics),
            ItemKind::Fn(i) => Some(&i.generics),
            ItemKind::TyAlias(i) => Some(&i.generics),
            ItemKind::TraitAlias(generics, _)
            | ItemKind::Enum(_, generics)
            | ItemKind::Struct(_, generics)
            | ItemKind::Union(_, generics) => Some(&generics),
            ItemKind::Trait(i) => Some(&i.generics),
            ItemKind::Impl(i) => Some(&i.generics),
        }
    }
}

/// `extern` qualifier on a function item or function type.
#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub enum Extern {
    /// No explicit extern keyword was used.
    ///
    /// E.g. `fn foo() {}`.
    None,
    /// An explicit extern keyword was used, but with implicit ABI.
    ///
    /// E.g. `extern fn foo() {}`.
    ///
    /// This is just `extern "C"` (see `rustc_target::spec::abi::Abi::FALLBACK`).
    Implicit(Span),
    /// An explicit extern keyword was used with an explicit ABI.
    ///
    /// E.g. `extern "C" fn foo() {}`.
    Explicit(StrLit, Span),
}

impl Extern {
    pub fn from_abi(abi: Option<StrLit>, span: Span) -> Extern {
        match abi {
            Some(name) => Extern::Explicit(name, span),
            None => Extern::Implicit(span),
        }
    }
}

/// A function header.
///
/// All the information between the visibility and the name of the function is
/// included in this struct (e.g., `async unsafe fn` or `const extern "C" fn`).
#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub struct FnHeader {
    /// Whether this is `unsafe`, or has a default safety.
    pub safety: Safety,
    /// Whether this is `async`, `gen`, or nothing.
    pub coroutine_kind: Option<CoroutineKind>,
    /// The `const` keyword, if any
    pub constness: Const,
    /// The `extern` keyword and corresponding ABI string, if any.
    pub ext: Extern,
}

impl FnHeader {
    /// Does this function header have any qualifiers or is it empty?
    pub fn has_qualifiers(&self) -> bool {
        let Self { safety, coroutine_kind, constness, ext } = self;
        matches!(safety, Safety::Unsafe(_))
            || coroutine_kind.is_some()
            || matches!(constness, Const::Yes(_))
            || !matches!(ext, Extern::None)
    }
}

impl Default for FnHeader {
    fn default() -> FnHeader {
        FnHeader {
            safety: Safety::Default,
            coroutine_kind: None,
            constness: Const::No,
            ext: Extern::None,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Trait {
    pub safety: Safety,
    pub is_auto: IsAuto,
    pub generics: Generics,
    pub bounds: GenericBounds,
    pub items: ThinVec<P<AssocItem>>,
}

/// The location of a where clause on a `TyAlias` (`Span`) and whether there was
/// a `where` keyword (`bool`). This is split out from `WhereClause`, since there
/// are two locations for where clause on type aliases, but their predicates
/// are concatenated together.
///
/// Take this example:
/// ```ignore (only-for-syntax-highlight)
/// trait Foo {
///   type Assoc<'a, 'b> where Self: 'a, Self: 'b;
/// }
/// impl Foo for () {
///   type Assoc<'a, 'b> where Self: 'a = () where Self: 'b;
///   //                 ^^^^^^^^^^^^^^ first where clause
///   //                                     ^^^^^^^^^^^^^^ second where clause
/// }
/// ```
///
/// If there is no where clause, then this is `false` with `DUMMY_SP`.
#[derive(Copy, Clone, Encodable, Decodable, Debug, Default)]
pub struct TyAliasWhereClause {
    pub has_where_token: bool,
    pub span: Span,
}

/// The span information for the two where clauses on a `TyAlias`.
#[derive(Copy, Clone, Encodable, Decodable, Debug, Default)]
pub struct TyAliasWhereClauses {
    /// Before the equals sign.
    pub before: TyAliasWhereClause,
    /// After the equals sign.
    pub after: TyAliasWhereClause,
    /// The index in `TyAlias.generics.where_clause.predicates` that would split
    /// into predicates from the where clause before the equals sign and the ones
    /// from the where clause after the equals sign.
    pub split: usize,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct TyAlias {
    pub defaultness: Defaultness,
    pub generics: Generics,
    pub where_clauses: TyAliasWhereClauses,
    pub bounds: GenericBounds,
    pub ty: Option<P<Ty>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Impl {
    pub defaultness: Defaultness,
    pub safety: Safety,
    pub generics: Generics,
    pub constness: Const,
    pub polarity: ImplPolarity,
    /// The trait being implemented, if any.
    pub of_trait: Option<TraitRef>,
    pub self_ty: P<Ty>,
    pub items: ThinVec<P<AssocItem>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Fn {
    pub defaultness: Defaultness,
    pub generics: Generics,
    pub sig: FnSig,
    pub body: Option<P<Block>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Delegation {
    /// Path resolution id.
    pub id: NodeId,
    pub qself: Option<P<QSelf>>,
    pub path: Path,
    pub rename: Option<Ident>,
    pub body: Option<P<Block>>,
    /// The item was expanded from a glob delegation item.
    pub from_glob: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct DelegationMac {
    pub qself: Option<P<QSelf>>,
    pub prefix: Path,
    // Some for list delegation, and None for glob delegation.
    pub suffixes: Option<ThinVec<(Ident, Option<Ident>)>>,
    pub body: Option<P<Block>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct StaticItem {
    pub ty: P<Ty>,
    pub safety: Safety,
    pub mutability: Mutability,
    pub expr: Option<P<Expr>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ConstItem {
    pub defaultness: Defaultness,
    pub generics: Generics,
    pub ty: P<Ty>,
    pub expr: Option<P<Expr>>,
}

// Adding a new variant? Please update `test_item` in `tests/ui/macros/stringify.rs`.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ItemKind {
    /// An `extern crate` item, with the optional *original* crate name if the crate was renamed.
    ///
    /// E.g., `extern crate foo` or `extern crate foo_bar as foo`.
    ExternCrate(Option<Symbol>),
    /// A use declaration item (`use`).
    ///
    /// E.g., `use foo;`, `use foo::bar;` or `use foo::bar as FooBar;`.
    Use(UseTree),
    /// A static item (`static`).
    ///
    /// E.g., `static FOO: i32 = 42;` or `static FOO: &'static str = "bar";`.
    Static(Box<StaticItem>),
    /// A constant item (`const`).
    ///
    /// E.g., `const FOO: i32 = 42;`.
    Const(Box<ConstItem>),
    /// A function declaration (`fn`).
    ///
    /// E.g., `fn foo(bar: usize) -> usize { .. }`.
    Fn(Box<Fn>),
    /// A module declaration (`mod`).
    ///
    /// E.g., `mod foo;` or `mod foo { .. }`.
    /// `unsafe` keyword on modules is accepted syntactically for macro DSLs, but not
    /// semantically by Rust.
    Mod(Safety, ModKind),
    /// An external module (`extern`).
    ///
    /// E.g., `extern {}` or `extern "C" {}`.
    ForeignMod(ForeignMod),
    /// Module-level inline assembly (from `global_asm!()`).
    GlobalAsm(Box<InlineAsm>),
    /// A type alias (`type`).
    ///
    /// E.g., `type Foo = Bar<u8>;`.
    TyAlias(Box<TyAlias>),
    /// An enum definition (`enum`).
    ///
    /// E.g., `enum Foo<A, B> { C<A>, D<B> }`.
    Enum(EnumDef, Generics),
    /// A struct definition (`struct`).
    ///
    /// E.g., `struct Foo<A> { x: A }`.
    Struct(VariantData, Generics),
    /// A union definition (`union`).
    ///
    /// E.g., `union Foo<A, B> { x: A, y: B }`.
    Union(VariantData, Generics),
    /// A trait declaration (`trait`).
    ///
    /// E.g., `trait Foo { .. }`, `trait Foo<T> { .. }` or `auto trait Foo {}`.
    Trait(Box<Trait>),
    /// Trait alias.
    ///
    /// E.g., `trait Foo = Bar + Quux;`.
    TraitAlias(Generics, GenericBounds),
    /// An implementation.
    ///
    /// E.g., `impl<A> Foo<A> { .. }` or `impl<A> Trait for Foo<A> { .. }`.
    Impl(Box<Impl>),
    /// A macro invocation.
    ///
    /// E.g., `foo!(..)`.
    MacCall(P<MacCall>),

    /// A macro definition.
    MacroDef(MacroDef),

    /// A single delegation item (`reuse`).
    ///
    /// E.g. `reuse <Type as Trait>::name { target_expr_template }`.
    Delegation(Box<Delegation>),
    /// A list or glob delegation item (`reuse prefix::{a, b, c}`, `reuse prefix::*`).
    /// Treated similarly to a macro call and expanded early.
    DelegationMac(Box<DelegationMac>),
}

impl ItemKind {
    /// "a" or "an"
    pub fn article(&self) -> &'static str {
        use ItemKind::*;
        match self {
            Use(..) | Static(..) | Const(..) | Fn(..) | Mod(..) | GlobalAsm(..) | TyAlias(..)
            | Struct(..) | Union(..) | Trait(..) | TraitAlias(..) | MacroDef(..)
            | Delegation(..) | DelegationMac(..) => "a",
            ExternCrate(..) | ForeignMod(..) | MacCall(..) | Enum(..) | Impl { .. } => "an",
        }
    }

    pub fn descr(&self) -> &'static str {
        match self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "`use` import",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod(..) => "extern block",
            ItemKind::GlobalAsm(..) => "global asm item",
            ItemKind::TyAlias(..) => "type alias",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::MacCall(..) => "item macro invocation",
            ItemKind::MacroDef(..) => "macro definition",
            ItemKind::Impl { .. } => "implementation",
            ItemKind::Delegation(..) => "delegated function",
            ItemKind::DelegationMac(..) => "delegation",
        }
    }

    pub fn generics(&self) -> Option<&Generics> {
        match self {
            Self::Fn(box Fn { generics, .. })
            | Self::TyAlias(box TyAlias { generics, .. })
            | Self::Const(box ConstItem { generics, .. })
            | Self::Enum(_, generics)
            | Self::Struct(_, generics)
            | Self::Union(_, generics)
            | Self::Trait(box Trait { generics, .. })
            | Self::TraitAlias(generics, _)
            | Self::Impl(box Impl { generics, .. }) => Some(generics),
            _ => None,
        }
    }
}

/// Represents associated items.
/// These include items in `impl` and `trait` definitions.
pub type AssocItem = Item<AssocItemKind>;

/// Represents associated item kinds.
///
/// The term "provided" in the variants below refers to the item having a default
/// definition / body. Meanwhile, a "required" item lacks a definition / body.
/// In an implementation, all items must be provided.
/// The `Option`s below denote the bodies, where `Some(_)`
/// means "provided" and conversely `None` means "required".
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AssocItemKind {
    /// An associated constant, `const $ident: $ty $def?;` where `def ::= "=" $expr? ;`.
    /// If `def` is parsed, then the constant is provided, and otherwise required.
    Const(Box<ConstItem>),
    /// An associated function.
    Fn(Box<Fn>),
    /// An associated type.
    Type(Box<TyAlias>),
    /// A macro expanding to associated items.
    MacCall(P<MacCall>),
    /// An associated delegation item.
    Delegation(Box<Delegation>),
    /// An associated list or glob delegation item.
    DelegationMac(Box<DelegationMac>),
}

impl AssocItemKind {
    pub fn defaultness(&self) -> Defaultness {
        match *self {
            Self::Const(box ConstItem { defaultness, .. })
            | Self::Fn(box Fn { defaultness, .. })
            | Self::Type(box TyAlias { defaultness, .. }) => defaultness,
            Self::MacCall(..) | Self::Delegation(..) | Self::DelegationMac(..) => {
                Defaultness::Final
            }
        }
    }
}

impl From<AssocItemKind> for ItemKind {
    fn from(assoc_item_kind: AssocItemKind) -> ItemKind {
        match assoc_item_kind {
            AssocItemKind::Const(item) => ItemKind::Const(item),
            AssocItemKind::Fn(fn_kind) => ItemKind::Fn(fn_kind),
            AssocItemKind::Type(ty_alias_kind) => ItemKind::TyAlias(ty_alias_kind),
            AssocItemKind::MacCall(a) => ItemKind::MacCall(a),
            AssocItemKind::Delegation(delegation) => ItemKind::Delegation(delegation),
            AssocItemKind::DelegationMac(delegation) => ItemKind::DelegationMac(delegation),
        }
    }
}

impl TryFrom<ItemKind> for AssocItemKind {
    type Error = ItemKind;

    fn try_from(item_kind: ItemKind) -> Result<AssocItemKind, ItemKind> {
        Ok(match item_kind {
            ItemKind::Const(item) => AssocItemKind::Const(item),
            ItemKind::Fn(fn_kind) => AssocItemKind::Fn(fn_kind),
            ItemKind::TyAlias(ty_kind) => AssocItemKind::Type(ty_kind),
            ItemKind::MacCall(a) => AssocItemKind::MacCall(a),
            ItemKind::Delegation(d) => AssocItemKind::Delegation(d),
            ItemKind::DelegationMac(d) => AssocItemKind::DelegationMac(d),
            _ => return Err(item_kind),
        })
    }
}

/// An item in `extern` block.
#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ForeignItemKind {
    /// A foreign static item (`static FOO: u8`).
    Static(Box<StaticItem>),
    /// An foreign function.
    Fn(Box<Fn>),
    /// An foreign type.
    TyAlias(Box<TyAlias>),
    /// A macro expanding to foreign items.
    MacCall(P<MacCall>),
}

impl From<ForeignItemKind> for ItemKind {
    fn from(foreign_item_kind: ForeignItemKind) -> ItemKind {
        match foreign_item_kind {
            ForeignItemKind::Static(box static_foreign_item) => {
                ItemKind::Static(Box::new(static_foreign_item.into()))
            }
            ForeignItemKind::Fn(fn_kind) => ItemKind::Fn(fn_kind),
            ForeignItemKind::TyAlias(ty_alias_kind) => ItemKind::TyAlias(ty_alias_kind),
            ForeignItemKind::MacCall(a) => ItemKind::MacCall(a),
        }
    }
}

impl TryFrom<ItemKind> for ForeignItemKind {
    type Error = ItemKind;

    fn try_from(item_kind: ItemKind) -> Result<ForeignItemKind, ItemKind> {
        Ok(match item_kind {
            ItemKind::Static(box static_item) => {
                ForeignItemKind::Static(Box::new(static_item.into()))
            }
            ItemKind::Fn(fn_kind) => ForeignItemKind::Fn(fn_kind),
            ItemKind::TyAlias(ty_alias_kind) => ForeignItemKind::TyAlias(ty_alias_kind),
            ItemKind::MacCall(a) => ForeignItemKind::MacCall(a),
            _ => return Err(item_kind),
        })
    }
}

pub type ForeignItem = Item<ForeignItemKind>;

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(AssocItem, 88);
    static_assert_size!(AssocItemKind, 16);
    static_assert_size!(Attribute, 32);
    static_assert_size!(Block, 32);
    static_assert_size!(Expr, 72);
    static_assert_size!(ExprKind, 40);
    static_assert_size!(Fn, 160);
    static_assert_size!(ForeignItem, 88);
    static_assert_size!(ForeignItemKind, 16);
    static_assert_size!(GenericArg, 24);
    static_assert_size!(GenericBound, 88);
    static_assert_size!(Generics, 40);
    static_assert_size!(Impl, 136);
    static_assert_size!(Item, 136);
    static_assert_size!(ItemKind, 64);
    static_assert_size!(LitKind, 24);
    static_assert_size!(Local, 80);
    static_assert_size!(MetaItemLit, 40);
    static_assert_size!(Param, 40);
    static_assert_size!(Pat, 72);
    static_assert_size!(Path, 24);
    static_assert_size!(PathSegment, 24);
    static_assert_size!(PatKind, 48);
    static_assert_size!(Stmt, 32);
    static_assert_size!(StmtKind, 16);
    static_assert_size!(Ty, 64);
    static_assert_size!(TyKind, 40);
    // tidy-alphabetical-end
}
