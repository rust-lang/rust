// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The Rust abstract syntax tree.

pub use self::TyParamBound::*;
pub use self::UnsafeSource::*;
pub use self::PathParameters::*;
pub use symbol::{Ident, Symbol as Name};
pub use util::ThinVec;
pub use util::parser::ExprPrecedence;

use syntax_pos::{Span, DUMMY_SP};
use codemap::{respan, Spanned};
use rustc_target::spec::abi::Abi;
use ext::hygiene::{Mark, SyntaxContext};
use print::pprust;
use ptr::P;
use rustc_data_structures::indexed_vec;
use symbol::{Symbol, keywords};
use tokenstream::{ThinTokenStream, TokenStream};

use serialize::{self, Encoder, Decoder};
use std::collections::HashSet;
use std::fmt;
use rustc_data_structures::sync::Lrc;
use std::u32;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct Label {
    pub ident: Ident,
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "label({:?})", self.ident)
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct Lifetime {
    pub id: NodeId,
    pub ident: Ident,
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

/// A lifetime definition, e.g. `'a: 'b+'c+'d`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct LifetimeDef {
    pub attrs: ThinVec<Attribute>,
    pub lifetime: Lifetime,
    pub bounds: Vec<Lifetime>
}

/// A "Path" is essentially Rust's notion of a name.
///
/// It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
///
/// E.g. `std::cmp::PartialEq`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Path {
    pub span: Span,
    /// The segments in the path: the things separated by `::`.
    /// Global paths begin with `keywords::CrateRoot`.
    pub segments: Vec<PathSegment>,
}

impl<'a> PartialEq<&'a str> for Path {
    fn eq(&self, string: &&'a str) -> bool {
        self.segments.len() == 1 && self.segments[0].ident.name == *string
    }
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "path({})", pprust::path_to_string(self))
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", pprust::path_to_string(self))
    }
}

impl Path {
    // convert a span and an identifier to the corresponding
    // 1-segment path
    pub fn from_ident(ident: Ident) -> Path {
        Path { segments: vec![PathSegment::from_ident(ident)], span: ident.span }
    }

    // Make a "crate root" segment for this path unless it already has it
    // or starts with something like `self`/`super`/`$crate`/etc.
    pub fn make_root(&self) -> Option<PathSegment> {
        if let Some(ident) = self.segments.get(0).map(|seg| seg.ident) {
            if ::parse::token::is_path_segment_keyword(ident) &&
               ident.name != keywords::Crate.name() {
                return None;
            }
        }
        Some(PathSegment::crate_root(self.span.shrink_to_lo()))
    }

    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == keywords::CrateRoot.name()
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of types.
///
/// E.g. `std`, `String` or `Box<T>`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub ident: Ident,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`.
    /// `None` means that no parameter list is supplied (`Path`),
    /// `Some` means that parameter list is supplied (`Path<X, Y>`)
    /// but it can be empty (`Path<>`).
    /// `P` is used as a size optimization for the common case with no parameters.
    pub parameters: Option<P<PathParameters>>,
}

impl PathSegment {
    pub fn from_ident(ident: Ident) -> Self {
        PathSegment { ident, parameters: None }
    }
    pub fn crate_root(span: Span) -> Self {
        PathSegment::from_ident(Ident::new(keywords::CrateRoot.name(), span))
    }
}

/// Parameters of a path segment.
///
/// E.g. `<A, B>` as in `Foo<A, B>` or `(A, B)` as in `Foo(A, B)`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum PathParameters {
    /// The `<'a, A,B,C>` in `foo::bar::baz::<'a, A,B,C>`
    AngleBracketed(AngleBracketedParameterData),
    /// The `(A,B)` and `C` in `Foo(A,B) -> C`
    Parenthesized(ParenthesizedParameterData),
}

impl PathParameters {
    pub fn span(&self) -> Span {
        match *self {
            AngleBracketed(ref data) => data.span,
            Parenthesized(ref data) => data.span,
        }
    }
}

/// A path like `Foo<'a, T>`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Default)]
pub struct AngleBracketedParameterData {
    /// Overall span
    pub span: Span,
    /// The lifetime parameters for this path segment.
    pub lifetimes: Vec<Lifetime>,
    /// The type parameters for this path segment, if present.
    pub types: Vec<P<Ty>>,
    /// Bindings (equality constraints) on associated types, if present.
    ///
    /// E.g., `Foo<A=Bar>`.
    pub bindings: Vec<TypeBinding>,
}

impl Into<Option<P<PathParameters>>> for AngleBracketedParameterData {
    fn into(self) -> Option<P<PathParameters>> {
        Some(P(PathParameters::AngleBracketed(self)))
    }
}

impl Into<Option<P<PathParameters>>> for ParenthesizedParameterData {
    fn into(self) -> Option<P<PathParameters>> {
        Some(P(PathParameters::Parenthesized(self)))
    }
}

/// A path like `Foo(A,B) -> C`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ParenthesizedParameterData {
    /// Overall span
    pub span: Span,

    /// `(A,B)`
    pub inputs: Vec<P<Ty>>,

    /// `C`
    pub output: Option<P<Ty>>,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodeId(u32);

impl NodeId {
    pub fn new(x: usize) -> NodeId {
        assert!(x < (u32::MAX as usize));
        NodeId(x as u32)
    }

    pub fn from_u32(x: u32) -> NodeId {
        NodeId(x)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn placeholder_from_mark(mark: Mark) -> Self {
        NodeId(mark.as_u32())
    }

    pub fn placeholder_to_mark(self) -> Mark {
        Mark::from_u32(self.0)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl serialize::UseSpecializedEncodable for NodeId {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.0)
    }
}

impl serialize::UseSpecializedDecodable for NodeId {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<NodeId, D::Error> {
        d.read_u32().map(NodeId)
    }
}

impl indexed_vec::Idx for NodeId {
    fn new(idx: usize) -> Self {
        NodeId::new(idx)
    }

    fn index(self) -> usize {
        self.as_usize()
    }
}

/// Node id used to represent the root of the crate.
pub const CRATE_NODE_ID: NodeId = NodeId(0);

/// When parsing and doing expansions, we initially give all AST nodes this AST
/// node value. Then later, in the renumber pass, we renumber them to have
/// small, positive ids.
pub const DUMMY_NODE_ID: NodeId = NodeId(!0);

/// The AST represents all type param bounds as types.
/// typeck::collect::compute_bounds matches these against
/// the "special" built-in traits (see middle::lang_items) and
/// detects Copy, Send and Sync.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyParamBound {
    TraitTyParamBound(PolyTraitRef, TraitBoundModifier),
    RegionTyParamBound(Lifetime)
}

impl TyParamBound {
    pub fn span(&self) -> Span {
        match self {
            &TraitTyParamBound(ref t, ..) => t.span,
            &RegionTyParamBound(ref l) => l.ident.span,
        }
    }
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

pub type TyParamBounds = Vec<TyParamBound>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TyParam {
    pub attrs: ThinVec<Attribute>,
    pub ident: Ident,
    pub id: NodeId,
    pub bounds: TyParamBounds,
    pub default: Option<P<Ty>>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum GenericParam {
    Lifetime(LifetimeDef),
    Type(TyParam),
}

impl GenericParam {
    pub fn is_lifetime_param(&self) -> bool {
        match *self {
            GenericParam::Lifetime(_) => true,
            _ => false,
        }
    }

    pub fn is_type_param(&self) -> bool {
        match *self {
            GenericParam::Type(_) => true,
            _ => false,
        }
    }
}

/// Represents lifetime, type and const parameters attached to a declaration of
/// a function, enum, trait, etc.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Generics {
    pub fn is_lt_parameterized(&self) -> bool {
        self.params.iter().any(|param| param.is_lifetime_param())
    }

    pub fn is_type_parameterized(&self) -> bool {
        self.params.iter().any(|param| param.is_type_param())
    }

    pub fn is_parameterized(&self) -> bool {
        !self.params.is_empty()
    }

    pub fn span_for_name(&self, name: &str) -> Option<Span> {
        for param in &self.params {
            if let GenericParam::Type(ref t) = *param {
                if t.ident.name == name {
                    return Some(t.ident.span);
                }
            }
        }
        None
    }
}

impl Default for Generics {
    /// Creates an instance of `Generics`.
    fn default() ->  Generics {
        Generics {
            params: Vec::new(),
            where_clause: WhereClause {
                id: DUMMY_NODE_ID,
                predicates: Vec::new(),
                span: DUMMY_SP,
            },
            span: DUMMY_SP,
        }
    }
}

/// A `where` clause in a definition
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereClause {
    pub id: NodeId,
    pub predicates: Vec<WherePredicate>,
    pub span: Span,
}

/// A single predicate in a `where` clause
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum WherePredicate {
    /// A type binding, e.g. `for<'c> Foo: Send+Clone+'c`
    BoundPredicate(WhereBoundPredicate),
    /// A lifetime predicate, e.g. `'a: 'b+'c`
    RegionPredicate(WhereRegionPredicate),
    /// An equality predicate (unsupported)
    EqPredicate(WhereEqPredicate),
}

impl WherePredicate {
    pub fn span(&self) -> Span {
        match self {
            &WherePredicate::BoundPredicate(ref p) => p.span,
            &WherePredicate::RegionPredicate(ref p) => p.span,
            &WherePredicate::EqPredicate(ref p) => p.span,
        }
    }
}

/// A type bound.
///
/// E.g. `for<'c> Foo: Send+Clone+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any generics from a `for` binding
    pub bound_generic_params: Vec<GenericParam>,
    /// The type being bounded
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (`Clone+Send+'static`)
    pub bounds: TyParamBounds,
}

/// A lifetime predicate.
///
/// E.g. `'a: 'b+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: Vec<Lifetime>,
}

/// An equality predicate (unsupported).
///
/// E.g. `T=int`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereEqPredicate {
    pub id: NodeId,
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

/// The set of MetaItems that define the compilation environment of the crate,
/// used to drive conditional compilation
pub type CrateConfig = HashSet<(Name, Option<Symbol>)>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Crate {
    pub module: Mod,
    pub attrs: Vec<Attribute>,
    pub span: Span,
}

/// A spanned compile-time attribute list item.
pub type NestedMetaItem = Spanned<NestedMetaItemKind>;

/// Possible values inside of compile-time attribute lists.
///
/// E.g. the '..' in `#[name(..)]`.
#[derive(Clone, Eq, RustcEncodable, RustcDecodable, Hash, Debug, PartialEq)]
pub enum NestedMetaItemKind {
    /// A full MetaItem, for recursive meta items.
    MetaItem(MetaItem),
    /// A literal.
    ///
    /// E.g. "foo", 64, true
    Literal(Lit),
}

/// A spanned compile-time attribute item.
///
/// E.g. `#[test]`, `#[derive(..)]`, `#[rustfmt::skip]` or `#[feature = "foo"]`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MetaItem {
    pub ident: Path,
    pub node: MetaItemKind,
    pub span: Span,
}

/// A compile-time attribute item.
///
/// E.g. `#[test]`, `#[derive(..)]` or `#[feature = "foo"]`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MetaItemKind {
    /// Word meta item.
    ///
    /// E.g. `test` as in `#[test]`
    Word,
    /// List meta item.
    ///
    /// E.g. `derive(..)` as in `#[derive(..)]`
    List(Vec<NestedMetaItem>),
    /// Name value meta item.
    ///
    /// E.g. `feature = "foo"` as in `#[feature = "foo"]`
    NameValue(Lit)
}

/// A Block (`{ .. }`).
///
/// E.g. `{ .. }` as in `fn foo() { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Block {
    /// Statements in a block
    pub stmts: Vec<Stmt>,
    pub id: NodeId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`
    pub rules: BlockCheckMode,
    pub span: Span,
    pub recovered: bool,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Pat {
    pub id: NodeId,
    pub node: PatKind,
    pub span: Span,
}

impl fmt::Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pat({}: {})", self.id, pprust::pat_to_string(self))
    }
}

impl Pat {
    pub(super) fn to_ty(&self) -> Option<P<Ty>> {
        let node = match &self.node {
            PatKind::Wild => TyKind::Infer,
            PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), ident, None) =>
                TyKind::Path(None, Path::from_ident(*ident)),
            PatKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            PatKind::Mac(mac) => TyKind::Mac(mac.clone()),
            PatKind::Ref(pat, mutbl) =>
                pat.to_ty().map(|ty| TyKind::Rptr(None, MutTy { ty, mutbl: *mutbl }))?,
            PatKind::Slice(pats, None, _) if pats.len() == 1 =>
                pats[0].to_ty().map(TyKind::Slice)?,
            PatKind::Tuple(pats, None) => {
                let mut tys = Vec::new();
                for pat in pats {
                    tys.push(pat.to_ty()?);
                }
                TyKind::Tup(tys)
            }
            _ => return None,
        };

        Some(P(Ty { node, id: self.id, span: self.span }))
    }

    pub fn walk<F>(&self, it: &mut F) -> bool
        where F: FnMut(&Pat) -> bool
    {
        if !it(self) {
            return false;
        }

        match self.node {
            PatKind::Ident(_, _, Some(ref p)) => p.walk(it),
            PatKind::Struct(_, ref fields, _) => {
                fields.iter().all(|field| field.node.pat.walk(it))
            }
            PatKind::TupleStruct(_, ref s, _) | PatKind::Tuple(ref s, _) => {
                s.iter().all(|p| p.walk(it))
            }
            PatKind::Box(ref s) | PatKind::Ref(ref s, _) | PatKind::Paren(ref s) => {
                s.walk(it)
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                before.iter().all(|p| p.walk(it)) &&
                slice.iter().all(|p| p.walk(it)) &&
                after.iter().all(|p| p.walk(it))
            }
            PatKind::Wild |
            PatKind::Lit(_) |
            PatKind::Range(..) |
            PatKind::Ident(..) |
            PatKind::Path(..) |
            PatKind::Mac(_) => {
                true
            }
        }
    }
}

/// A single field in a struct pattern
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except is_shorthand is true
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct FieldPat {
    /// The identifier for the field
    pub ident: Ident,
    /// The pattern the field is destructured to
    pub pat: P<Pat>,
    pub is_shorthand: bool,
    pub attrs: ThinVec<Attribute>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BindingMode {
    ByRef(Mutability),
    ByValue(Mutability),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum RangeEnd {
    Included(RangeSyntax),
    Excluded,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum RangeSyntax {
    DotDotDot,
    DotDotEq,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum PatKind {
    /// Represents a wildcard pattern (`_`)
    Wild,

    /// A `PatKind::Ident` may either be a new bound variable (`ref mut binding @ OPT_SUBPATTERN`),
    /// or a unit struct/variant pattern, or a const pattern (in the last two cases the third
    /// field must be `None`). Disambiguation cannot be done with parser alone, so it happens
    /// during name resolution.
    Ident(BindingMode, Ident, Option<P<Pat>>),

    /// A struct or struct variant pattern, e.g. `Variant {x, y, ..}`.
    /// The `bool` is `true` in the presence of a `..`.
    Struct(Path, Vec<Spanned<FieldPat>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    TupleStruct(Path, Vec<P<Pat>>, Option<usize>),

    /// A possibly qualified path pattern.
    /// Unqualified path patterns `A::B::C` can legally refer to variants, structs, constants
    /// or associated constants. Qualified path patterns `<A>::B::C`/`<A as Trait>::B::C` can
    /// only legally refer to associated constants.
    Path(Option<QSelf>, Path),

    /// A tuple pattern `(a, b)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    Tuple(Vec<P<Pat>>, Option<usize>),
    /// A `box` pattern
    Box(P<Pat>),
    /// A reference pattern, e.g. `&mut (a, b)`
    Ref(P<Pat>, Mutability),
    /// A literal
    Lit(P<Expr>),
    /// A range pattern, e.g. `1...2`, `1..=2` or `1..2`
    Range(P<Expr>, P<Expr>, RangeEnd),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Slice(box [a, b], Some(i), box [y, z])`
    Slice(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
    /// Parentheses in patters used for grouping, i.e. `(PAT)`.
    Paren(P<Pat>),
    /// A macro pattern; pre-expansion
    Mac(Mac),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
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
    pub fn to_string(&self) -> &'static str {
        use self::BinOpKind::*;
        match *self {
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
    pub fn lazy(&self) -> bool {
        match *self {
            BinOpKind::And | BinOpKind::Or => true,
            _ => false
        }
    }

    pub fn is_shift(&self) -> bool {
        match *self {
            BinOpKind::Shl | BinOpKind::Shr => true,
            _ => false
        }
    }

    pub fn is_comparison(&self) -> bool {
        use self::BinOpKind::*;
        match *self {
            Eq | Lt | Le | Ne | Gt | Ge =>
            true,
            And | Or | Add | Sub | Mul | Div | Rem |
            BitXor | BitAnd | BitOr | Shl | Shr =>
            false,
        }
    }

    /// Returns `true` if the binary operator takes its arguments by value
    pub fn is_by_value(&self) -> bool {
        !self.is_comparison()
    }
}

pub type BinOp = Spanned<BinOpKind>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum UnOp {
    /// The `*` operator for dereferencing
    Deref,
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl UnOp {
    /// Returns `true` if the unary operator takes its argument by value
    pub fn is_by_value(u: UnOp) -> bool {
        match u {
            UnOp::Neg | UnOp::Not => true,
            _ => false,
        }
    }

    pub fn to_string(op: UnOp) -> &'static str {
        match op {
            UnOp::Deref => "*",
            UnOp::Not => "!",
            UnOp::Neg => "-",
        }
    }
}

/// A statement
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Stmt {
    pub id: NodeId,
    pub node: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn add_trailing_semicolon(mut self) -> Self {
        self.node = match self.node {
            StmtKind::Expr(expr) => StmtKind::Semi(expr),
            StmtKind::Mac(mac) => StmtKind::Mac(mac.map(|(mac, _style, attrs)| {
                (mac, MacStmtStyle::Semicolon, attrs)
            })),
            node => node,
        };
        self
    }

    pub fn is_item(&self) -> bool {
        match self.node {
            StmtKind::Item(_) => true,
            _ => false,
        }
    }

    pub fn is_expr(&self) -> bool {
        match self.node {
            StmtKind::Expr(_) => true,
            _ => false,
        }
    }
}

impl fmt::Debug for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "stmt({}: {})", self.id.to_string(), pprust::stmt_to_string(self))
    }
}


#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum StmtKind {
    /// A local (let) binding.
    Local(P<Local>),

    /// An item definition.
    Item(P<Item>),

    /// Expr without trailing semi-colon.
    Expr(P<Expr>),
    /// Expr with a trailing semi-colon.
    Semi(P<Expr>),
    /// Macro.
    Mac(P<(Mac, MacStmtStyle, ThinVec<Attribute>)>),
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MacStmtStyle {
    /// The macro statement had a trailing semicolon, e.g. `foo! { ... };`
    /// `foo!(...);`, `foo![...];`
    Semicolon,
    /// The macro statement had braces; e.g. foo! { ... }
    Braces,
    /// The macro statement had parentheses or brackets and no semicolon; e.g.
    /// `foo!(...)`. All of these will end up being converted into macro
    /// expressions.
    NoBraces,
}

/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Local {
    pub pat: P<Pat>,
    pub ty: Option<P<Ty>>,
    /// Initializer expression to set the value, if any
    pub init: Option<P<Expr>>,
    pub id: NodeId,
    pub span: Span,
    pub attrs: ThinVec<Attribute>,
}

/// An arm of a 'match'.
///
/// E.g. `0...10 => { println!("match!") }` as in
///
/// ```
/// match 123 {
///     0...10 => { println!("match!") },
///     _ => { println!("no match!") },
/// }
/// ```
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arm {
    pub attrs: Vec<Attribute>,
    pub pats: Vec<P<Pat>>,
    pub guard: Option<P<Expr>>,
    pub body: P<Expr>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Field {
    pub ident: Ident,
    pub expr: P<Expr>,
    pub span: Span,
    pub is_shorthand: bool,
    pub attrs: ThinVec<Attribute>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BlockCheckMode {
    Default,
    Unsafe(UnsafeSource),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

/// An expression
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash,)]
pub struct Expr {
    pub id: NodeId,
    pub node: ExprKind,
    pub span: Span,
    pub attrs: ThinVec<Attribute>
}

impl Expr {
    /// Whether this expression would be valid somewhere that expects a value, for example, an `if`
    /// condition.
    pub fn returns(&self) -> bool {
        if let ExprKind::Block(ref block) = self.node {
            match block.stmts.last().map(|last_stmt| &last_stmt.node) {
                // implicit return
                Some(&StmtKind::Expr(_)) => true,
                Some(&StmtKind::Semi(ref expr)) => {
                    if let ExprKind::Ret(_) = expr.node {
                        // last statement is explicit return
                        true
                    } else {
                        false
                    }
                }
                // This is a block that doesn't end in either an implicit or explicit return
                _ => false,
            }
        } else {
            // This is not a block, it is a value
            true
        }
    }

    fn to_bound(&self) -> Option<TyParamBound> {
        match &self.node {
            ExprKind::Path(None, path) =>
                Some(TraitTyParamBound(PolyTraitRef::new(Vec::new(), path.clone(), self.span),
                                       TraitBoundModifier::None)),
            _ => None,
        }
    }

    pub(super) fn to_ty(&self) -> Option<P<Ty>> {
        let node = match &self.node {
            ExprKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            ExprKind::Mac(mac) => TyKind::Mac(mac.clone()),
            ExprKind::Paren(expr) => expr.to_ty().map(TyKind::Paren)?,
            ExprKind::AddrOf(mutbl, expr) =>
                expr.to_ty().map(|ty| TyKind::Rptr(None, MutTy { ty, mutbl: *mutbl }))?,
            ExprKind::Repeat(expr, expr_len) =>
                expr.to_ty().map(|ty| TyKind::Array(ty, expr_len.clone()))?,
            ExprKind::Array(exprs) if exprs.len() == 1 =>
                exprs[0].to_ty().map(TyKind::Slice)?,
            ExprKind::Tup(exprs) => {
                let mut tys = Vec::new();
                for expr in exprs {
                    tys.push(expr.to_ty()?);
                }
                TyKind::Tup(tys)
            }
            ExprKind::Binary(binop, lhs, rhs) if binop.node == BinOpKind::Add =>
                if let (Some(lhs), Some(rhs)) = (lhs.to_bound(), rhs.to_bound()) {
                    TyKind::TraitObject(vec![lhs, rhs], TraitObjectSyntax::None)
                } else {
                    return None;
                }
            _ => return None,
        };

        Some(P(Ty { node, id: self.id, span: self.span }))
    }

    pub fn precedence(&self) -> ExprPrecedence {
        match self.node {
            ExprKind::Box(_) => ExprPrecedence::Box,
            ExprKind::ObsoleteInPlace(..) => ExprPrecedence::ObsoleteInPlace,
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::If(..) => ExprPrecedence::If,
            ExprKind::IfLet(..) => ExprPrecedence::IfLet,
            ExprKind::While(..) => ExprPrecedence::While,
            ExprKind::WhileLet(..) => ExprPrecedence::WhileLet,
            ExprKind::ForLoop(..) => ExprPrecedence::ForLoop,
            ExprKind::Loop(..) => ExprPrecedence::Loop,
            ExprKind::Match(..) => ExprPrecedence::Match,
            ExprKind::Closure(..) => ExprPrecedence::Closure,
            ExprKind::Block(..) => ExprPrecedence::Block,
            ExprKind::Catch(..) => ExprPrecedence::Catch,
            ExprKind::Assign(..) => ExprPrecedence::Assign,
            ExprKind::AssignOp(..) => ExprPrecedence::AssignOp,
            ExprKind::Field(..) => ExprPrecedence::Field,
            ExprKind::Index(..) => ExprPrecedence::Index,
            ExprKind::Range(..) => ExprPrecedence::Range,
            ExprKind::Path(..) => ExprPrecedence::Path,
            ExprKind::AddrOf(..) => ExprPrecedence::AddrOf,
            ExprKind::Break(..) => ExprPrecedence::Break,
            ExprKind::Continue(..) => ExprPrecedence::Continue,
            ExprKind::Ret(..) => ExprPrecedence::Ret,
            ExprKind::InlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::Mac(..) => ExprPrecedence::Mac,
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Paren(..) => ExprPrecedence::Paren,
            ExprKind::Try(..) => ExprPrecedence::Try,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

/// Limit types of a range (inclusive or exclusive)
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum RangeLimits {
    /// Inclusive at the beginning, exclusive at the end
    HalfOpen,
    /// Inclusive at the beginning and end
    Closed,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ExprKind {
    /// A `box x` expression.
    Box(P<Expr>),
    /// First expr is the place; second expr is the value.
    ObsoleteInPlace(P<Expr>, P<Expr>),
    /// An array (`[a, b, c, d]`)
    Array(Vec<P<Expr>>),
    /// A function call
    ///
    /// The first field resolves to the function itself,
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(P<Expr>, Vec<P<Expr>>),
    /// A method call (`x.foo::<'static, Bar, Baz>(a, b, c, d)`)
    ///
    /// The `PathSegment` represents the method name and its generic arguments
    /// (within the angle brackets).
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, [x, a, b, c, d])`.
    MethodCall(PathSegment, Vec<P<Expr>>),
    /// A tuple (`(a, b, c ,d)`)
    Tup(Vec<P<Expr>>),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, P<Expr>),
    /// A literal (For example: `1`, `"foo"`)
    Lit(P<Lit>),
    /// A cast (`foo as f64`)
    Cast(P<Expr>, P<Ty>),
    Type(P<Expr>, P<Ty>),
    /// An `if` block, with an optional else block
    ///
    /// `if expr { block } else { expr }`
    If(P<Expr>, P<Block>, Option<P<Expr>>),
    /// An `if let` expression with an optional else block
    ///
    /// `if let pat = expr { block } else { expr }`
    ///
    /// This is desugared to a `match` expression.
    IfLet(Vec<P<Pat>>, P<Expr>, P<Block>, Option<P<Expr>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    While(P<Expr>, P<Block>, Option<Label>),
    /// A while-let loop, with an optional label
    ///
    /// `'label: while let pat = expr { block }`
    ///
    /// This is desugared to a combination of `loop` and `match` expressions.
    WhileLet(Vec<P<Pat>>, P<Expr>, P<Block>, Option<Label>),
    /// A for loop, with an optional label
    ///
    /// `'label: for pat in expr { block }`
    ///
    /// This is desugared to a combination of `loop` and `match` expressions.
    ForLoop(P<Pat>, P<Expr>, P<Block>, Option<Label>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    Loop(P<Block>, Option<Label>),
    /// A `match` block.
    Match(P<Expr>, Vec<Arm>),
    /// A closure (for example, `move |a, b, c| a + b + c`)
    ///
    /// The final span is the span of the argument block `|...|`
    Closure(CaptureBy, Movability, P<FnDecl>, P<Expr>, Span),
    /// A block (`{ ... }`)
    Block(P<Block>),
    /// A catch block (`catch { ... }`)
    Catch(P<Block>),

    /// An assignment (`a = foo()`)
    Assign(P<Expr>, P<Expr>),
    /// An assignment with an operator
    ///
    /// For example, `a += 1`.
    AssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct field
    Field(P<Expr>, Ident),
    /// An indexing operation (`foo[2]`)
    Index(P<Expr>, P<Expr>),
    /// A range (`1..2`, `1..`, `..2`, `1...2`, `1...`, `...2`)
    Range(Option<P<Expr>>, Option<P<Expr>>, RangeLimits),

    /// Variable reference, possibly containing `::` and/or type
    /// parameters, e.g. foo::bar::<baz>.
    ///
    /// Optionally "qualified",
    /// E.g. `<Vec<T> as SomeTrait>::SomeType`.
    Path(Option<QSelf>, Path),

    /// A referencing operation (`&a` or `&mut a`)
    AddrOf(Mutability, P<Expr>),
    /// A `break`, with an optional label to break, and an optional expression
    Break(Option<Label>, Option<P<Expr>>),
    /// A `continue`, with an optional label
    Continue(Option<Label>),
    /// A `return`, with an optional value to be returned
    Ret(Option<P<Expr>>),

    /// Output of the `asm!()` macro
    InlineAsm(P<InlineAsm>),

    /// A macro invocation; pre-expansion
    Mac(Mac),

    /// A struct literal expression.
    ///
    /// For example, `Foo {x: 1, y: 2}`, or
    /// `Foo {x: 1, .. base}`, where `base` is the `Option<Expr>`.
    Struct(Path, Vec<Field>, Option<P<Expr>>),

    /// An array literal constructed from one repeated element.
    ///
    /// For example, `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(P<Expr>, P<Expr>),

    /// No-op: used solely so we can pretty-print faithfully
    Paren(P<Expr>),

    /// `expr?`
    Try(P<Expr>),

    /// A `yield`, with an optional value to be yielded
    Yield(Option<P<Expr>>),
}

/// The explicit Self type in a "qualified path". The actual
/// path, including the trait and the associated item, is stored
/// separately. `position` represents the index of the associated
/// item qualified with this Self type.
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
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct QSelf {
    pub ty: P<Ty>,
    pub position: usize
}

/// A capture clause
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum CaptureBy {
    Value,
    Ref,
}

/// The movability of a generator / closure literal
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Movability {
    Static,
    Movable,
}

pub type Mac = Spanned<Mac_>;

/// Represents a macro invocation. The Path indicates which macro
/// is being invoked, and the vector of token-trees contains the source
/// of the macro invocation.
///
/// NB: the additional ident for a macro_rules-style macro is actually
/// stored in the enclosing item. Oog.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Mac_ {
    pub path: Path,
    pub tts: ThinTokenStream,
}

impl Mac_ {
    pub fn stream(&self) -> TokenStream {
        self.tts.clone().into()
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MacroDef {
    pub tokens: ThinTokenStream,
    pub legacy: bool,
}

impl MacroDef {
    pub fn stream(&self) -> TokenStream {
        self.tokens.clone().into()
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum StrStyle {
    /// A regular string, like `"foo"`
    Cooked,
    /// A raw string, like `r##"foo"##`
    ///
    /// The value is the number of `#` symbols used.
    Raw(u16)
}

/// A literal
pub type Lit = Spanned<LitKind>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum LitIntType {
    Signed(IntTy),
    Unsigned(UintTy),
    Unsuffixed,
}

/// Literal kind.
///
/// E.g. `"foo"`, `42`, `12.34` or `bool`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(Symbol, StrStyle),
    /// A byte string (`b"foo"`)
    ByteStr(Lrc<Vec<u8>>),
    /// A byte char (`b'f'`)
    Byte(u8),
    /// A character literal (`'a'`)
    Char(char),
    /// An integer literal (`1`)
    Int(u128, LitIntType),
    /// A float literal (`1f64` or `1E10f64`)
    Float(Symbol, FloatTy),
    /// A float literal without a suffix (`1.0 or 1.0E10`)
    FloatUnsuffixed(Symbol),
    /// A boolean literal
    Bool(bool),
}

impl LitKind {
    /// Returns true if this literal is a string and false otherwise.
    pub fn is_str(&self) -> bool {
        match *self {
            LitKind::Str(..) => true,
            _ => false,
        }
    }

    /// Returns true if this literal has no suffix. Note: this will return true
    /// for literals with prefixes such as raw strings and byte strings.
    pub fn is_unsuffixed(&self) -> bool {
        match *self {
            // unsuffixed variants
            LitKind::Str(..) |
            LitKind::ByteStr(..) |
            LitKind::Byte(..) |
            LitKind::Char(..) |
            LitKind::Int(_, LitIntType::Unsuffixed) |
            LitKind::FloatUnsuffixed(..) |
            LitKind::Bool(..) => true,
            // suffixed variants
            LitKind::Int(_, LitIntType::Signed(..)) |
            LitKind::Int(_, LitIntType::Unsigned(..)) |
            LitKind::Float(..) => false,
        }
    }

    /// Returns true if this literal has a suffix.
    pub fn is_suffixed(&self) -> bool {
        !self.is_unsuffixed()
    }
}

// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

/// Represents a method's signature in a trait declaration,
/// or in an implementation.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MethodSig {
    pub unsafety: Unsafety,
    pub constness: Spanned<Constness>,
    pub abi: Abi,
    pub decl: P<FnDecl>,
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitItem {
    pub id: NodeId,
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub generics: Generics,
    pub node: TraitItemKind,
    pub span: Span,
    /// See `Item::tokens` for what this is
    pub tokens: Option<TokenStream>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitItemKind {
    Const(P<Ty>, Option<P<Expr>>),
    Method(MethodSig, Option<P<Block>>),
    Type(TyParamBounds, Option<P<Ty>>),
    Macro(Mac),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ImplItem {
    pub id: NodeId,
    pub ident: Ident,
    pub vis: Visibility,
    pub defaultness: Defaultness,
    pub attrs: Vec<Attribute>,
    pub generics: Generics,
    pub node: ImplItemKind,
    pub span: Span,
    /// See `Item::tokens` for what this is
    pub tokens: Option<TokenStream>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ImplItemKind {
    Const(P<Ty>, P<Expr>),
    Method(MethodSig, P<Block>),
    Type(P<Ty>),
    Macro(Mac),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy,
         PartialOrd, Ord)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl fmt::Debug for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl IntTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn val_to_string(&self, val: i128) -> String {
        // cast to a u128 so we can correctly print INT128_MIN. All integral types
        // are parsed as u128, so we wouldn't want to print an extra negative
        // sign.
        format!("{}{}", val as u128, self.ty_to_string())
    }

    pub fn bit_width(&self) -> Option<usize> {
        Some(match *self {
            IntTy::Isize => return None,
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        })
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy,
         PartialOrd, Ord)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn val_to_string(&self, val: u128) -> String {
        format!("{}{}", val, self.ty_to_string())
    }

    pub fn bit_width(&self) -> Option<usize> {
        Some(match *self {
            UintTy::Usize => return None,
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        })
    }
}

impl fmt::Debug for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy,
         PartialOrd, Ord)]
pub enum FloatTy {
    F32,
    F64,
}

impl fmt::Debug for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl FloatTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    pub fn bit_width(&self) -> usize {
        match *self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
        }
    }
}

// Bind a type to an associated type: `A=Foo`.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TypeBinding {
    pub id: NodeId,
    pub ident: Ident,
    pub ty: P<Ty>,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Ty {
    pub id: NodeId,
    pub node: TyKind,
    pub span: Span,
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "type({})", pprust::ty_to_string(self))
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct BareFnTy {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub generic_params: Vec<GenericParam>,
    pub decl: P<FnDecl>
}

/// The different kinds of types recognized by the compiler
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyKind {
    /// A variable-length slice (`[T]`)
    Slice(P<Ty>),
    /// A fixed length array (`[T; n]`)
    Array(P<Ty>, P<Expr>),
    /// A raw pointer (`*const T` or `*mut T`)
    Ptr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`)
    Rptr(Option<Lifetime>, MutTy),
    /// A bare function (e.g. `fn(usize) -> bool`)
    BareFn(P<BareFnTy>),
    /// The never type (`!`)
    Never,
    /// A tuple (`(A, B, C, D,...)`)
    Tup(Vec<P<Ty>> ),
    /// A path (`module::module::...::Type`), optionally
    /// "qualified", e.g. `<Vec<T> as SomeTrait>::SomeType`.
    ///
    /// Type parameters are stored in the Path itself
    Path(Option<QSelf>, Path),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(TyParamBounds, TraitObjectSyntax),
    /// An `impl Bound1 + Bound2 + Bound3` type
    /// where `Bound` is a trait or a lifetime.
    ImplTrait(TyParamBounds),
    /// No-op; kept solely so that we can pretty-print faithfully
    Paren(P<Ty>),
    /// Unused for now
    Typeof(P<Expr>),
    /// TyKind::Infer means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Inferred type of a `self` or `&self` argument in a method.
    ImplicitSelf,
    // A macro in the type position.
    Mac(Mac),
    /// Placeholder for a kind that has failed to be defined.
    Err,
}

/// Syntax used to declare a trait object.
#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitObjectSyntax {
    Dyn,
    None,
}

/// Inline assembly dialect.
///
/// E.g. `"intel"` as in `asm!("mov eax, 2" : "={eax}"(result) : : : "intel")`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum AsmDialect {
    Att,
    Intel,
}

/// Inline assembly.
///
/// E.g. `"={eax}"(result)` as in `asm!("mov eax, 2" : "={eax}"(result) : : : "intel")`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsmOutput {
    pub constraint: Symbol,
    pub expr: P<Expr>,
    pub is_rw: bool,
    pub is_indirect: bool,
}

/// Inline assembly.
///
/// E.g. `asm!("NOP");`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsm {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<InlineAsmOutput>,
    pub inputs: Vec<(Symbol, P<Expr>)>,
    pub clobbers: Vec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    pub ctxt: SyntaxContext,
}

/// An argument in a function header.
///
/// E.g. `bar: usize` as in `fn foo(bar: usize)`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arg {
    pub ty: P<Ty>,
    pub pat: P<Pat>,
    pub id: NodeId,
}

/// Alternative representation for `Arg`s describing `self` parameter of methods.
///
/// E.g. `&mut self` as in `fn foo(&mut self)`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum SelfKind {
    /// `self`, `mut self`
    Value(Mutability),
    /// `&'lt self`, `&'lt mut self`
    Region(Option<Lifetime>, Mutability),
    /// `self: TYPE`, `mut self: TYPE`
    Explicit(P<Ty>, Mutability),
}

pub type ExplicitSelf = Spanned<SelfKind>;

impl Arg {
    pub fn to_self(&self) -> Option<ExplicitSelf> {
        if let PatKind::Ident(BindingMode::ByValue(mutbl), ident, _) = self.pat.node {
            if ident.name == keywords::SelfValue.name() {
                return match self.ty.node {
                    TyKind::ImplicitSelf => Some(respan(self.pat.span, SelfKind::Value(mutbl))),
                    TyKind::Rptr(lt, MutTy{ref ty, mutbl}) if ty.node == TyKind::ImplicitSelf => {
                        Some(respan(self.pat.span, SelfKind::Region(lt, mutbl)))
                    }
                    _ => Some(respan(self.pat.span.to(self.ty.span),
                                     SelfKind::Explicit(self.ty.clone(), mutbl))),
                }
            }
        }
        None
    }

    pub fn is_self(&self) -> bool {
        if let PatKind::Ident(_, ident, _) = self.pat.node {
            ident.name == keywords::SelfValue.name()
        } else {
            false
        }
    }

    pub fn from_self(eself: ExplicitSelf, eself_ident: Ident) -> Arg {
        let span = eself.span.to(eself_ident.span);
        let infer_ty = P(Ty {
            id: DUMMY_NODE_ID,
            node: TyKind::ImplicitSelf,
            span,
        });
        let arg = |mutbl, ty| Arg {
            pat: P(Pat {
                id: DUMMY_NODE_ID,
                node: PatKind::Ident(BindingMode::ByValue(mutbl), eself_ident, None),
                span,
            }),
            ty,
            id: DUMMY_NODE_ID,
        };
        match eself.node {
            SelfKind::Explicit(ty, mutbl) => arg(mutbl, ty),
            SelfKind::Value(mutbl) => arg(mutbl, infer_ty),
            SelfKind::Region(lt, mutbl) => arg(Mutability::Immutable, P(Ty {
                id: DUMMY_NODE_ID,
                node: TyKind::Rptr(lt, MutTy { ty: infer_ty, mutbl: mutbl }),
                span,
            })),
        }
    }
}

/// Header (not the body) of a function declaration.
///
/// E.g. `fn foo(bar: baz)`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct FnDecl {
    pub inputs: Vec<Arg>,
    pub output: FunctionRetTy,
    pub variadic: bool
}

impl FnDecl {
    pub fn get_self(&self) -> Option<ExplicitSelf> {
        self.inputs.get(0).and_then(Arg::to_self)
    }
    pub fn has_self(&self) -> bool {
        self.inputs.get(0).map(Arg::is_self).unwrap_or(false)
    }
}

/// Is the trait definition an auto trait?
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum IsAuto {
    Yes,
    No
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Unsafety {
    Unsafe,
    Normal,
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Constness {
    Const,
    NotConst,
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Defaultness {
    Default,
    Final,
}

impl fmt::Display for Unsafety {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(match *self {
            Unsafety::Normal => "normal",
            Unsafety::Unsafe => "unsafe",
        }, f)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ImplPolarity::Positive => "positive".fmt(f),
            ImplPolarity::Negative => "negative".fmt(f),
        }
    }
}


#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum FunctionRetTy {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    Default(Span),
    /// Everything else
    Ty(P<Ty>),
}

impl FunctionRetTy {
    pub fn span(&self) -> Span {
        match *self {
            FunctionRetTy::Default(span) => span,
            FunctionRetTy::Ty(ref ty) => ty.span,
        }
    }
}

/// Module declaration.
///
/// E.g. `mod foo;` or `mod foo { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub items: Vec<P<Item>>,
}

/// Foreign module declaration.
///
/// E.g. `extern { .. }` or `extern C { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignMod {
    pub abi: Abi,
    pub items: Vec<ForeignItem>,
}

/// Global inline assembly
///
/// aka module-level assembly or file-scoped assembly
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub struct GlobalAsm {
    pub asm: Symbol,
    pub ctxt: SyntaxContext,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct EnumDef {
    pub variants: Vec<Variant>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Variant_ {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub data: VariantData,
    /// Explicit discriminant, e.g. `Foo = 1`
    pub disr_expr: Option<P<Expr>>,
}

pub type Variant = Spanned<Variant_>;

/// Part of `use` item to the right of its prefix.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum UseTreeKind {
    /// `use prefix` or `use prefix as rename`
    Simple(Option<Ident>),
    /// `use prefix::{...}`
    Nested(Vec<(UseTree, NodeId)>),
    /// `use prefix::*`
    Glob,
}

/// A tree of paths sharing common prefixes.
/// Used in `use` items both at top-level and inside of braces in import groups.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct UseTree {
    pub prefix: Path,
    pub kind: UseTreeKind,
    pub span: Span,
}

impl UseTree {
    pub fn ident(&self) -> Ident {
        match self.kind {
            UseTreeKind::Simple(Some(rename)) => rename,
            UseTreeKind::Simple(None) =>
                self.prefix.segments.last().expect("empty prefix in a simple import").ident,
            _ => panic!("`UseTree::ident` can only be used on a simple import"),
        }
    }
}

/// Distinguishes between Attributes that decorate items and Attributes that
/// are contained as statements within items. These two cases need to be
/// distinguished for pretty-printing.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum AttrStyle {
    Outer,
    Inner,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub struct AttrId(pub usize);

/// Meta-data associated with an item
/// Doc-comments are promoted to attributes that have is_sugared_doc = true
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Attribute {
    pub id: AttrId,
    pub style: AttrStyle,
    pub path: Path,
    pub tokens: TokenStream,
    pub is_sugared_doc: bool,
    pub span: Span,
}

/// TraitRef's appear in impls.
///
/// resolve maps each TraitRef's ref_id to its defining trait; that's all
/// that the ref_id is for. The impl_id maps to the "self type" of this impl.
/// If this impl is an ItemKind::Impl, the impl_id is redundant (it could be the
/// same as the impl's node id).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PolyTraitRef {
    /// The `'a` in `<'a> Foo<&'a T>`
    pub bound_generic_params: Vec<GenericParam>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef,

    pub span: Span,
}

impl PolyTraitRef {
    pub fn new(generic_params: Vec<GenericParam>, path: Path, span: Span) -> Self {
        PolyTraitRef {
            bound_generic_params: generic_params,
            trait_ref: TraitRef { path: path, ref_id: DUMMY_NODE_ID },
            span,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum CrateSugar {
    /// Source is `pub(crate)`
    PubCrate,

    /// Source is (just) `crate`
    JustCrate,
}

pub type Visibility = Spanned<VisibilityKind>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum VisibilityKind {
    Public,
    Crate(CrateSugar),
    Restricted { path: P<Path>, id: NodeId },
    Inherited,
}

/// Field of a struct.
///
/// E.g. `bar: usize` as in `struct Foo { bar: usize }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct StructField {
    pub span: Span,
    pub ident: Option<Ident>,
    pub vis: Visibility,
    pub id: NodeId,
    pub ty: P<Ty>,
    pub attrs: Vec<Attribute>,
}

/// Fields and Ids of enum variants and structs
///
/// For enum variants: `NodeId` represents both an Id of the variant itself (relevant for all
/// variant kinds) and an Id of the variant's constructor (not relevant for `Struct`-variants).
/// One shared Id can be successfully used for these two purposes.
/// Id of the whole enum lives in `Item`.
///
/// For structs: `NodeId` represents an Id of the structure's constructor, so it is not actually
/// used for `Struct`-structs (but still presents). Structures don't have an analogue of "Id of
/// the variant itself" from enum variants.
/// Id of the whole struct lives in `Item`.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum VariantData {
    /// Struct variant.
    ///
    /// E.g. `Bar { .. }` as in `enum Foo { Bar { .. } }`
    Struct(Vec<StructField>, NodeId),
    /// Tuple variant.
    ///
    /// E.g. `Bar(..)` as in `enum Foo { Bar(..) }`
    Tuple(Vec<StructField>, NodeId),
    /// Unit variant.
    ///
    /// E.g. `Bar = ..` as in `enum Foo { Bar = .. }`
    Unit(NodeId),
}

impl VariantData {
    pub fn fields(&self) -> &[StructField] {
        match *self {
            VariantData::Struct(ref fields, _) | VariantData::Tuple(ref fields, _) => fields,
            _ => &[],
        }
    }
    pub fn id(&self) -> NodeId {
        match *self {
            VariantData::Struct(_, id) | VariantData::Tuple(_, id) | VariantData::Unit(id) => id
        }
    }
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self { true } else { false }
    }
    pub fn is_tuple(&self) -> bool {
        if let VariantData::Tuple(..) = *self { true } else { false }
    }
    pub fn is_unit(&self) -> bool {
        if let VariantData::Unit(..) = *self { true } else { false }
    }
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Item {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub id: NodeId,
    pub node: ItemKind,
    pub vis: Visibility,
    pub span: Span,

    /// Original tokens this item was parsed from. This isn't necessarily
    /// available for all items, although over time more and more items should
    /// have this be `Some`. Right now this is primarily used for procedural
    /// macros, notably custom attributes.
    ///
    /// Note that the tokens here do not include the outer attributes, but will
    /// include inner attributes.
    pub tokens: Option<TokenStream>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ItemKind {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// E.g. `extern crate foo` or `extern crate foo_bar as foo`
    ExternCrate(Option<Name>),
    /// A use declaration (`use` or `pub use`) item.
    ///
    /// E.g. `use foo;`, `use foo::bar;` or `use foo::bar as FooBar;`
    Use(P<UseTree>),
    /// A static item (`static` or `pub static`).
    ///
    /// E.g. `static FOO: i32 = 42;` or `static FOO: &'static str = "bar";`
    Static(P<Ty>, Mutability, P<Expr>),
    /// A constant item (`const` or `pub const`).
    ///
    /// E.g. `const FOO: i32 = 42;`
    Const(P<Ty>, P<Expr>),
    /// A function declaration (`fn` or `pub fn`).
    ///
    /// E.g. `fn foo(bar: usize) -> usize { .. }`
    Fn(P<FnDecl>, Unsafety, Spanned<Constness>, Abi, Generics, P<Block>),
    /// A module declaration (`mod` or `pub mod`).
    ///
    /// E.g. `mod foo;` or `mod foo { .. }`
    Mod(Mod),
    /// An external module (`extern` or `pub extern`).
    ///
    /// E.g. `extern {}` or `extern "C" {}`
    ForeignMod(ForeignMod),
    /// Module-level inline assembly (from `global_asm!()`)
    GlobalAsm(P<GlobalAsm>),
    /// A type alias (`type` or `pub type`).
    ///
    /// E.g. `type Foo = Bar<u8>;`
    Ty(P<Ty>, Generics),
    /// An enum definition (`enum` or `pub enum`).
    ///
    /// E.g. `enum Foo<A, B> { C<A>, D<B> }`
    Enum(EnumDef, Generics),
    /// A struct definition (`struct` or `pub struct`).
    ///
    /// E.g. `struct Foo<A> { x: A }`
    Struct(VariantData, Generics),
    /// A union definition (`union` or `pub union`).
    ///
    /// E.g. `union Foo<A, B> { x: A, y: B }`
    Union(VariantData, Generics),
    /// A Trait declaration (`trait` or `pub trait`).
    ///
    /// E.g. `trait Foo { .. }`, `trait Foo<T> { .. }` or `auto trait Foo {}`
    Trait(IsAuto, Unsafety, Generics, TyParamBounds, Vec<TraitItem>),
    /// Trait alias
    ///
    /// E.g. `trait Foo = Bar + Quux;`
    TraitAlias(Generics, TyParamBounds),
    /// An implementation.
    ///
    /// E.g. `impl<A> Foo<A> { .. }` or `impl<A> Trait for Foo<A> { .. }`
    Impl(Unsafety,
             ImplPolarity,
             Defaultness,
             Generics,
             Option<TraitRef>, // (optional) trait this impl implements
             P<Ty>, // self
             Vec<ImplItem>),
    /// A macro invocation.
    ///
    /// E.g. `macro_rules! foo { .. }` or `foo!(..)`
    Mac(Mac),

    /// A macro definition.
    MacroDef(MacroDef),
}

impl ItemKind {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "use",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod(..) => "foreign module",
            ItemKind::GlobalAsm(..) => "global asm",
            ItemKind::Ty(..) => "type alias",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::Mac(..) |
            ItemKind::MacroDef(..) |
            ItemKind::Impl(..) => "item"
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignItem {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub node: ForeignItemKind,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

/// An item within an `extern` block
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ForeignItemKind {
    /// A foreign function
    Fn(P<FnDecl>, Generics),
    /// A foreign static item (`static ext: u8`), with optional mutability
    /// (the boolean is true when mutable)
    Static(P<Ty>, bool),
    /// A foreign type
    Ty,
    /// A macro invocation
    Macro(Mac),
}

impl ForeignItemKind {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ForeignItemKind::Fn(..) => "foreign function",
            ForeignItemKind::Static(..) => "foreign static item",
            ForeignItemKind::Ty => "foreign type",
            ForeignItemKind::Macro(..) => "macro in foreign module",
        }
    }
}

#[cfg(test)]
mod tests {
    use serialize;
    use super::*;

    // are ASTs encodable?
    #[test]
    fn check_asts_encodable() {
        fn assert_encodable<T: serialize::Encodable>() {}
        assert_encodable::<Crate>();
    }
}
