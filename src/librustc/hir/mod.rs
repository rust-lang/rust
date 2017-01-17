// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The Rust HIR.

pub use self::BindingMode::*;
pub use self::BinOp_::*;
pub use self::BlockCheckMode::*;
pub use self::CaptureClause::*;
pub use self::Decl_::*;
pub use self::Expr_::*;
pub use self::FunctionRetTy::*;
pub use self::ForeignItem_::*;
pub use self::Item_::*;
pub use self::Mutability::*;
pub use self::PrimTy::*;
pub use self::Stmt_::*;
pub use self::Ty_::*;
pub use self::TyParamBound::*;
pub use self::UnOp::*;
pub use self::UnsafeSource::*;
pub use self::Visibility::{Public, Inherited};
pub use self::PathParameters::*;

use hir::def::Def;
use hir::def_id::DefId;
use util::nodemap::{NodeMap, FxHashMap, FxHashSet};

use syntax_pos::{Span, ExpnId, DUMMY_SP};
use syntax::codemap::{self, Spanned};
use syntax::abi::Abi;
use syntax::ast::{Name, NodeId, DUMMY_NODE_ID, AsmDialect};
use syntax::ast::{Attribute, Lit, StrStyle, FloatTy, IntTy, UintTy, MetaItem};
use syntax::ptr::P;
use syntax::symbol::{Symbol, keywords};
use syntax::tokenstream::TokenTree;
use syntax::util::ThinVec;

use std::collections::BTreeMap;
use std::fmt;

/// HIR doesn't commit to a concrete storage type and have its own alias for a vector.
/// It can be `Vec`, `P<[T]>` or potentially `Box<[T]>`, or some other container with similar
/// behavior. Unlike AST, HIR is mostly a static structure, so we can use an owned slice instead
/// of `Vec` to avoid keeping extra capacity.
pub type HirVec<T> = P<[T]>;

macro_rules! hir_vec {
    ($elem:expr; $n:expr) => (
        $crate::hir::HirVec::from(vec![$elem; $n])
    );
    ($($x:expr),*) => (
        $crate::hir::HirVec::from(vec![$($x),*])
    );
    ($($x:expr,)*) => (hir_vec![$($x),*])
}

pub mod check_attr;
pub mod def;
pub mod def_id;
pub mod intravisit;
pub mod itemlikevisit;
pub mod lowering;
pub mod map;
pub mod pat_util;
pub mod print;
pub mod svh;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct Lifetime {
    pub id: NodeId,
    pub span: Span,
    pub name: Name,
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "lifetime({}: {})",
               self.id,
               print::to_string(print::NO_ANN, |s| s.print_lifetime(self)))
    }
}

/// A lifetime definition, eg `'a: 'b+'c+'d`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct LifetimeDef {
    pub lifetime: Lifetime,
    pub bounds: HirVec<Lifetime>,
    pub pure_wrt_drop: bool,
}

/// A "Path" is essentially Rust's notion of a name; for instance:
/// std::cmp::PartialEq  .  It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Path {
    pub span: Span,
    /// The definition that the path resolved to.
    pub def: Def,
    /// The segments in the path: the things separated by `::`.
    pub segments: HirVec<PathSegment>,
}

impl Path {
    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].name == keywords::CrateRoot.name()
    }
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "path({})",
               print::to_string(print::NO_ANN, |s| s.print_path(self, false)))
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub name: Name,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`. Note that
    /// this is more than just simple syntactic sugar; the use of
    /// parens affects the region binding rules, so we preserve the
    /// distinction.
    pub parameters: PathParameters,
}

impl PathSegment {
    /// Convert an identifier to the corresponding segment.
    pub fn from_name(name: Name) -> PathSegment {
        PathSegment {
            name: name,
            parameters: PathParameters::none()
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum PathParameters {
    /// The `<'a, A,B,C>` in `foo::bar::baz::<'a, A,B,C>`
    AngleBracketedParameters(AngleBracketedParameterData),
    /// The `(A,B)` and `C` in `Foo(A,B) -> C`
    ParenthesizedParameters(ParenthesizedParameterData),
}

impl PathParameters {
    pub fn none() -> PathParameters {
        AngleBracketedParameters(AngleBracketedParameterData {
            lifetimes: HirVec::new(),
            types: HirVec::new(),
            infer_types: true,
            bindings: HirVec::new(),
        })
    }

    pub fn is_empty(&self) -> bool {
        match *self {
            AngleBracketedParameters(ref data) => data.is_empty(),

            // Even if the user supplied no types, something like
            // `X()` is equivalent to `X<(),()>`.
            ParenthesizedParameters(..) => false,
        }
    }

    pub fn has_lifetimes(&self) -> bool {
        match *self {
            AngleBracketedParameters(ref data) => !data.lifetimes.is_empty(),
            ParenthesizedParameters(_) => false,
        }
    }

    pub fn has_types(&self) -> bool {
        match *self {
            AngleBracketedParameters(ref data) => !data.types.is_empty(),
            ParenthesizedParameters(..) => true,
        }
    }

    /// Returns the types that the user wrote. Note that these do not necessarily map to the type
    /// parameters in the parenthesized case.
    pub fn types(&self) -> HirVec<&P<Ty>> {
        match *self {
            AngleBracketedParameters(ref data) => {
                data.types.iter().collect()
            }
            ParenthesizedParameters(ref data) => {
                data.inputs
                    .iter()
                    .chain(data.output.iter())
                    .collect()
            }
        }
    }

    pub fn lifetimes(&self) -> HirVec<&Lifetime> {
        match *self {
            AngleBracketedParameters(ref data) => {
                data.lifetimes.iter().collect()
            }
            ParenthesizedParameters(_) => {
                HirVec::new()
            }
        }
    }

    pub fn bindings(&self) -> HirVec<&TypeBinding> {
        match *self {
            AngleBracketedParameters(ref data) => {
                data.bindings.iter().collect()
            }
            ParenthesizedParameters(_) => {
                HirVec::new()
            }
        }
    }
}

/// A path like `Foo<'a, T>`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct AngleBracketedParameterData {
    /// The lifetime parameters for this path segment.
    pub lifetimes: HirVec<Lifetime>,
    /// The type parameters for this path segment, if present.
    pub types: HirVec<P<Ty>>,
    /// Whether to infer remaining type parameters, if any.
    /// This only applies to expression and pattern paths, and
    /// out of those only the segments with no type parameters
    /// to begin with, e.g. `Vec::new` is `<Vec<..>>::new::<..>`.
    pub infer_types: bool,
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A=Bar>`.
    pub bindings: HirVec<TypeBinding>,
}

impl AngleBracketedParameterData {
    fn is_empty(&self) -> bool {
        self.lifetimes.is_empty() && self.types.is_empty() && self.bindings.is_empty()
    }
}

/// A path like `Foo(A,B) -> C`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ParenthesizedParameterData {
    /// Overall span
    pub span: Span,

    /// `(A,B)`
    pub inputs: HirVec<P<Ty>>,

    /// `C`
    pub output: Option<P<Ty>>,
}

/// The AST represents all type param bounds as types.
/// typeck::collect::compute_bounds matches these against
/// the "special" built-in traits (see middle::lang_items) and
/// detects Copy, Send and Sync.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyParamBound {
    TraitTyParamBound(PolyTraitRef, TraitBoundModifier),
    RegionTyParamBound(Lifetime),
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

pub type TyParamBounds = HirVec<TyParamBound>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TyParam {
    pub name: Name,
    pub id: NodeId,
    pub bounds: TyParamBounds,
    pub default: Option<P<Ty>>,
    pub span: Span,
    pub pure_wrt_drop: bool,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Generics {
    pub lifetimes: HirVec<LifetimeDef>,
    pub ty_params: HirVec<TyParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Generics {
    pub fn empty() -> Generics {
        Generics {
            lifetimes: HirVec::new(),
            ty_params: HirVec::new(),
            where_clause: WhereClause {
                id: DUMMY_NODE_ID,
                predicates: HirVec::new(),
            },
            span: DUMMY_SP,
        }
    }

    pub fn is_lt_parameterized(&self) -> bool {
        !self.lifetimes.is_empty()
    }

    pub fn is_type_parameterized(&self) -> bool {
        !self.ty_params.is_empty()
    }

    pub fn is_parameterized(&self) -> bool {
        self.is_lt_parameterized() || self.is_type_parameterized()
    }
}

pub enum UnsafeGeneric {
    Region(LifetimeDef, &'static str),
    Type(TyParam, &'static str),
}

impl UnsafeGeneric {
    pub fn attr_name(&self) -> &'static str {
        match *self {
            UnsafeGeneric::Region(_, s) => s,
            UnsafeGeneric::Type(_, s) => s,
        }
    }
}

impl Generics {
    pub fn carries_unsafe_attr(&self) -> Option<UnsafeGeneric> {
        for r in &self.lifetimes {
            if r.pure_wrt_drop {
                return Some(UnsafeGeneric::Region(r.clone(), "may_dangle"));
            }
        }
        for t in &self.ty_params {
            if t.pure_wrt_drop {
                return Some(UnsafeGeneric::Type(t.clone(), "may_dangle"));
            }
        }
        return None;
    }
}

/// A `where` clause in a definition
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereClause {
    pub id: NodeId,
    pub predicates: HirVec<WherePredicate>,
}

/// A single predicate in a `where` clause
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum WherePredicate {
    /// A type binding, eg `for<'c> Foo: Send+Clone+'c`
    BoundPredicate(WhereBoundPredicate),
    /// A lifetime predicate, e.g. `'a: 'b+'c`
    RegionPredicate(WhereRegionPredicate),
    /// An equality predicate (unsupported)
    EqPredicate(WhereEqPredicate),
}

/// A type bound, eg `for<'c> Foo: Send+Clone+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any lifetimes from a `for` binding
    pub bound_lifetimes: HirVec<LifetimeDef>,
    /// The type being bounded
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (`Clone+Send+'static`)
    pub bounds: TyParamBounds,
}

/// A lifetime predicate, e.g. `'a: 'b+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: HirVec<Lifetime>,
}

/// An equality predicate (unsupported), e.g. `T=int`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereEqPredicate {
    pub id: NodeId,
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

pub type CrateConfig = HirVec<P<MetaItem>>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
pub struct Crate {
    pub module: Mod,
    pub attrs: HirVec<Attribute>,
    pub span: Span,
    pub exported_macros: HirVec<MacroDef>,

    // NB: We use a BTreeMap here so that `visit_all_items` iterates
    // over the ids in increasing order. In principle it should not
    // matter what order we visit things in, but in *practice* it
    // does, because it can affect the order in which errors are
    // detected, which in turn can make compile-fail tests yield
    // slightly different results.
    pub items: BTreeMap<NodeId, Item>,

    pub trait_items: BTreeMap<TraitItemId, TraitItem>,
    pub impl_items: BTreeMap<ImplItemId, ImplItem>,
    pub bodies: FxHashMap<BodyId, Body>,
}

impl Crate {
    pub fn item(&self, id: NodeId) -> &Item {
        &self.items[&id]
    }

    pub fn trait_item(&self, id: TraitItemId) -> &TraitItem {
        &self.trait_items[&id]
    }

    pub fn impl_item(&self, id: ImplItemId) -> &ImplItem {
        &self.impl_items[&id]
    }

    /// Visits all items in the crate in some determinstic (but
    /// unspecified) order. If you just need to process every item,
    /// but don't care about nesting, this method is the best choice.
    ///
    /// If you do care about nesting -- usually because your algorithm
    /// follows lexical scoping rules -- then you want a different
    /// approach. You should override `visit_nested_item` in your
    /// visitor and then call `intravisit::walk_crate` instead.
    pub fn visit_all_item_likes<'hir, V>(&'hir self, visitor: &mut V)
        where V: itemlikevisit::ItemLikeVisitor<'hir>
    {
        for (_, item) in &self.items {
            visitor.visit_item(item);
        }

        for (_, trait_item) in &self.trait_items {
            visitor.visit_trait_item(trait_item);
        }

        for (_, impl_item) in &self.impl_items {
            visitor.visit_impl_item(impl_item);
        }
    }

    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[&id]
    }
}

/// A macro definition, in this crate or imported from another.
///
/// Not parsed directly, but created on macro import or `macro_rules!` expansion.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MacroDef {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub id: NodeId,
    pub span: Span,
    pub body: HirVec<TokenTree>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Block {
    /// Statements in a block
    pub stmts: HirVec<Stmt>,
    /// An expression at the end of the block
    /// without a semicolon, if any
    pub expr: Option<P<Expr>>,
    pub id: NodeId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`
    pub rules: BlockCheckMode,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Pat {
    pub id: NodeId,
    pub node: PatKind,
    pub span: Span,
}

impl fmt::Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pat({}: {})", self.id,
               print::to_string(print::NO_ANN, |s| s.print_pat(self)))
    }
}

impl Pat {
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_<G>(&self, it: &mut G) -> bool
        where G: FnMut(&Pat) -> bool
    {
        if !it(self) {
            return false;
        }

        match self.node {
            PatKind::Binding(.., Some(ref p)) => p.walk_(it),
            PatKind::Struct(_, ref fields, _) => {
                fields.iter().all(|field| field.node.pat.walk_(it))
            }
            PatKind::TupleStruct(_, ref s, _) | PatKind::Tuple(ref s, _) => {
                s.iter().all(|p| p.walk_(it))
            }
            PatKind::Box(ref s) | PatKind::Ref(ref s, _) => {
                s.walk_(it)
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                before.iter().all(|p| p.walk_(it)) &&
                slice.iter().all(|p| p.walk_(it)) &&
                after.iter().all(|p| p.walk_(it))
            }
            PatKind::Wild |
            PatKind::Lit(_) |
            PatKind::Range(..) |
            PatKind::Binding(..) |
            PatKind::Path(_) => {
                true
            }
        }
    }

    pub fn walk<F>(&self, mut it: F) -> bool
        where F: FnMut(&Pat) -> bool
    {
        self.walk_(&mut it)
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
    pub name: Name,
    /// The pattern the field is destructured to
    pub pat: P<Pat>,
    pub is_shorthand: bool,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BindingMode {
    BindByRef(Mutability),
    BindByValue(Mutability),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum PatKind {
    /// Represents a wildcard pattern (`_`)
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `DefId` is for the definition of the variable being bound.
    Binding(BindingMode, DefId, Spanned<Name>, Option<P<Pat>>),

    /// A struct or struct variant pattern, e.g. `Variant {x, y, ..}`.
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath, HirVec<Spanned<FieldPat>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    TupleStruct(QPath, HirVec<P<Pat>>, Option<usize>),

    /// A path pattern for an unit struct/variant or a (maybe-associated) constant.
    Path(QPath),

    /// A tuple pattern `(a, b)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    Tuple(HirVec<P<Pat>>, Option<usize>),
    /// A `box` pattern
    Box(P<Pat>),
    /// A reference pattern, e.g. `&mut (a, b)`
    Ref(P<Pat>, Mutability),
    /// A literal
    Lit(P<Expr>),
    /// A range pattern, e.g. `1...2`
    Range(P<Expr>, P<Expr>),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Slice(box [a, b], Some(i), box [y, z])`
    Slice(HirVec<P<Pat>>, Option<P<Pat>>, HirVec<P<Pat>>),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BinOp_ {
    /// The `+` operator (addition)
    BiAdd,
    /// The `-` operator (subtraction)
    BiSub,
    /// The `*` operator (multiplication)
    BiMul,
    /// The `/` operator (division)
    BiDiv,
    /// The `%` operator (modulus)
    BiRem,
    /// The `&&` operator (logical and)
    BiAnd,
    /// The `||` operator (logical or)
    BiOr,
    /// The `^` operator (bitwise xor)
    BiBitXor,
    /// The `&` operator (bitwise and)
    BiBitAnd,
    /// The `|` operator (bitwise or)
    BiBitOr,
    /// The `<<` operator (shift left)
    BiShl,
    /// The `>>` operator (shift right)
    BiShr,
    /// The `==` operator (equality)
    BiEq,
    /// The `<` operator (less than)
    BiLt,
    /// The `<=` operator (less than or equal to)
    BiLe,
    /// The `!=` operator (not equal to)
    BiNe,
    /// The `>=` operator (greater than or equal to)
    BiGe,
    /// The `>` operator (greater than)
    BiGt,
}

impl BinOp_ {
    pub fn as_str(self) -> &'static str {
        match self {
            BiAdd => "+",
            BiSub => "-",
            BiMul => "*",
            BiDiv => "/",
            BiRem => "%",
            BiAnd => "&&",
            BiOr => "||",
            BiBitXor => "^",
            BiBitAnd => "&",
            BiBitOr => "|",
            BiShl => "<<",
            BiShr => ">>",
            BiEq => "==",
            BiLt => "<",
            BiLe => "<=",
            BiNe => "!=",
            BiGe => ">=",
            BiGt => ">",
        }
    }

    pub fn is_lazy(self) -> bool {
        match self {
            BiAnd | BiOr => true,
            _ => false,
        }
    }

    pub fn is_shift(self) -> bool {
        match self {
            BiShl | BiShr => true,
            _ => false,
        }
    }

    pub fn is_comparison(self) -> bool {
        match self {
            BiEq | BiLt | BiLe | BiNe | BiGt | BiGe => true,
            BiAnd |
            BiOr |
            BiAdd |
            BiSub |
            BiMul |
            BiDiv |
            BiRem |
            BiBitXor |
            BiBitAnd |
            BiBitOr |
            BiShl |
            BiShr => false,
        }
    }

    /// Returns `true` if the binary operator takes its arguments by value
    pub fn is_by_value(self) -> bool {
        !self.is_comparison()
    }
}

pub type BinOp = Spanned<BinOp_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum UnOp {
    /// The `*` operator for dereferencing
    UnDeref,
    /// The `!` operator for logical inversion
    UnNot,
    /// The `-` operator for negation
    UnNeg,
}

impl UnOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnDeref => "*",
            UnNot => "!",
            UnNeg => "-",
        }
    }

    /// Returns `true` if the unary operator takes its argument by value
    pub fn is_by_value(self) -> bool {
        match self {
            UnNeg | UnNot => true,
            _ => false,
        }
    }
}

/// A statement
pub type Stmt = Spanned<Stmt_>;

impl fmt::Debug for Stmt_ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Sadness.
        let spanned = codemap::dummy_spanned(self.clone());
        write!(f,
               "stmt({}: {})",
               spanned.node.id(),
               print::to_string(print::NO_ANN, |s| s.print_stmt(&spanned)))
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum Stmt_ {
    /// Could be an item or a local (let) binding:
    StmtDecl(P<Decl>, NodeId),

    /// Expr without trailing semi-colon (must have unit type):
    StmtExpr(P<Expr>, NodeId),

    /// Expr with trailing semi-colon (may have any type):
    StmtSemi(P<Expr>, NodeId),
}

impl Stmt_ {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtDecl(ref d, _) => d.node.attrs(),
            StmtExpr(ref e, _) |
            StmtSemi(ref e, _) => &e.attrs,
        }
    }

    pub fn id(&self) -> NodeId {
        match *self {
            StmtDecl(_, id) => id,
            StmtExpr(_, id) => id,
            StmtSemi(_, id) => id,
        }
    }
}

// FIXME (pending discussion of #1697, #2178...): local should really be
// a refinement on pat.
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

pub type Decl = Spanned<Decl_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Decl_ {
    /// A local (let) binding:
    DeclLocal(P<Local>),
    /// An item binding:
    DeclItem(ItemId),
}

impl Decl_ {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            DeclLocal(ref l) => &l.attrs,
            DeclItem(_) => &[]
        }
    }
}

/// represents one arm of a 'match'
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arm {
    pub attrs: HirVec<Attribute>,
    pub pats: HirVec<P<Pat>>,
    pub guard: Option<P<Expr>>,
    pub body: P<Expr>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Field {
    pub name: Spanned<Name>,
    pub expr: P<Expr>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
    PushUnsafeBlock(UnsafeSource),
    PopUnsafeBlock(UnsafeSource),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct BodyId {
    pub node_id: NodeId,
}

/// The body of a function or constant value.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Body {
    pub arguments: HirVec<Arg>,
    pub value: Expr
}

impl Body {
    pub fn id(&self) -> BodyId {
        BodyId {
            node_id: self.value.id
        }
    }
}

/// An expression
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Expr {
    pub id: NodeId,
    pub span: Span,
    pub node: Expr_,
    pub attrs: ThinVec<Attribute>,
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "expr({}: {})", self.id,
               print::to_string(print::NO_ANN, |s| s.print_expr(self)))
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Expr_ {
    /// A `box x` expression.
    ExprBox(P<Expr>),
    /// An array (`[a, b, c, d]`)
    ExprArray(HirVec<Expr>),
    /// A function call
    ///
    /// The first field resolves to the function itself (usually an `ExprPath`),
    /// and the second field is the list of arguments
    ExprCall(P<Expr>, HirVec<Expr>),
    /// A method call (`x.foo::<Bar, Baz>(a, b, c, d)`)
    ///
    /// The `Spanned<Name>` is the identifier for the method name.
    /// The vector of `Ty`s are the ascripted type parameters for the method
    /// (within the angle brackets).
    ///
    /// The first element of the vector of `Expr`s is the expression that
    /// evaluates to the object on which the method is being called on (the
    /// receiver), and the remaining elements are the rest of the arguments.
    ///
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprMethodCall(foo, [Bar, Baz], [x, a, b, c, d])`.
    ExprMethodCall(Spanned<Name>, HirVec<P<Ty>>, HirVec<Expr>),
    /// A tuple (`(a, b, c ,d)`)
    ExprTup(HirVec<Expr>),
    /// A binary operation (For example: `a + b`, `a * b`)
    ExprBinary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (For example: `!x`, `*x`)
    ExprUnary(UnOp, P<Expr>),
    /// A literal (For example: `1`, `"foo"`)
    ExprLit(P<Lit>),
    /// A cast (`foo as f64`)
    ExprCast(P<Expr>, P<Ty>),
    ExprType(P<Expr>, P<Ty>),
    /// An `if` block, with an optional else block
    ///
    /// `if expr { block } else { expr }`
    ExprIf(P<Expr>, P<Block>, Option<P<Expr>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    ExprWhile(P<Expr>, P<Block>, Option<Spanned<Name>>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    ExprLoop(P<Block>, Option<Spanned<Name>>, LoopSource),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    ExprMatch(P<Expr>, HirVec<Arm>, MatchSource),
    /// A closure (for example, `move |a, b, c| {a + b + c}`).
    ///
    /// The final span is the span of the argument block `|...|`
    ExprClosure(CaptureClause, P<FnDecl>, BodyId, Span),
    /// A block (`{ ... }`)
    ExprBlock(P<Block>),

    /// An assignment (`a = foo()`)
    ExprAssign(P<Expr>, P<Expr>),
    /// An assignment with an operator
    ///
    /// For example, `a += 1`.
    ExprAssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named struct field (`obj.foo`)
    ExprField(P<Expr>, Spanned<Name>),
    /// Access of an unnamed field of a struct or tuple-struct
    ///
    /// For example, `foo.0`.
    ExprTupField(P<Expr>, Spanned<usize>),
    /// An indexing operation (`foo[2]`)
    ExprIndex(P<Expr>, P<Expr>),

    /// Path to a definition, possibly containing lifetime or type parameters.
    ExprPath(QPath),

    /// A referencing operation (`&a` or `&mut a`)
    ExprAddrOf(Mutability, P<Expr>),
    /// A `break`, with an optional label to break
    ExprBreak(Option<Label>, Option<P<Expr>>),
    /// A `continue`, with an optional label
    ExprAgain(Option<Label>),
    /// A `return`, with an optional value to be returned
    ExprRet(Option<P<Expr>>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    ExprInlineAsm(P<InlineAsm>, HirVec<Expr>, HirVec<Expr>),

    /// A struct or struct-like variant literal expression.
    ///
    /// For example, `Foo {x: 1, y: 2}`, or
    /// `Foo {x: 1, .. base}`, where `base` is the `Option<Expr>`.
    ExprStruct(QPath, HirVec<Field>, Option<P<Expr>>),

    /// An array literal constructed from one repeated element.
    ///
    /// For example, `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    ExprRepeat(P<Expr>, BodyId),
}

/// Optionally `Self`-qualified value/type path or associated extension.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum QPath {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// E.g. an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<P<Ty>>, P<Path>),

    /// Type-related paths, e.g. `<T>::default` or `<T>::Output`.
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes being each a `TyPath(QPath::TypeRelative(..))`.
    TypeRelative(P<Ty>, P<PathSegment>)
}

/// Hints at the original code for a `match _ { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum MatchSource {
    /// A `match _ { .. }`
    Normal,
    /// An `if let _ = _ { .. }` (optionally with `else { .. }`)
    IfLetDesugar {
        contains_else_clause: bool,
    },
    /// A `while let _ = _ { .. }` (which was desugared to a
    /// `loop { match _ { .. } }`)
    WhileLetDesugar,
    /// A desugared `for _ in _ { .. }` loop
    ForLoopDesugar,
    /// A desugared `?` operator
    TryDesugar,
}

/// The loop type that yielded an ExprLoop
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum LoopSource {
    /// A `loop { .. }` loop
    Loop,
    /// A `while let _ = _ { .. }` loop
    WhileLet,
    /// A `for _ in _ { .. }` loop
    ForLoop,
}


#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub struct Label {
    pub span: Span,
    pub name: Name,
    pub loop_id: NodeId
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum CaptureClause {
    CaptureByValue,
    CaptureByRef,
}

// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

/// Represents a method's signature in a trait declaration or implementation.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MethodSig {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub abi: Abi,
    pub decl: P<FnDecl>,
    pub generics: Generics,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitItemId {
    pub node_id: NodeId,
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitItem {
    pub id: NodeId,
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub node: TraitItemKind,
    pub span: Span,
}

/// A trait method's body (or just argument names).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitMethod {
    /// No default body in the trait, just a signature.
    Required(HirVec<Spanned<Name>>),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitItemKind {
    /// An associated constant with an optional value (otherwise `impl`s
    /// must contain a value)
    Const(P<Ty>, Option<BodyId>),
    /// A method with an optional body
    Method(MethodSig, TraitMethod),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type
    Type(TyParamBounds, Option<P<Ty>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ImplItemId {
    pub node_id: NodeId,
}

/// Represents anything within an `impl` block
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ImplItem {
    pub id: NodeId,
    pub name: Name,
    pub vis: Visibility,
    pub defaultness: Defaultness,
    pub attrs: HirVec<Attribute>,
    pub node: ImplItemKind,
    pub span: Span,
}

/// Represents different contents within `impl`s
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ImplItemKind {
    /// An associated constant of the given type, set to the constant result
    /// of the expression
    Const(P<Ty>, BodyId),
    /// A method implementation with the given signature and body
    Method(MethodSig, BodyId),
    /// An associated type
    Type(P<Ty>),
}

// Bind a type to an associated type: `A=Foo`.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TypeBinding {
    pub id: NodeId,
    pub name: Name,
    pub ty: P<Ty>,
    pub span: Span,
}


#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Ty {
    pub id: NodeId,
    pub node: Ty_,
    pub span: Span,
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "type({})",
               print::to_string(print::NO_ANN, |s| s.print_type(self)))
    }
}

/// Not represented directly in the AST, referred to by name through a ty_path.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum PrimTy {
    TyInt(IntTy),
    TyUint(UintTy),
    TyFloat(FloatTy),
    TyStr,
    TyBool,
    TyChar,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct BareFnTy {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub lifetimes: HirVec<LifetimeDef>,
    pub decl: P<FnDecl>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
/// The different kinds of types recognized by the compiler
pub enum Ty_ {
    /// A variable length slice (`[T]`)
    TySlice(P<Ty>),
    /// A fixed length array (`[T; n]`)
    TyArray(P<Ty>, BodyId),
    /// A raw pointer (`*const T` or `*mut T`)
    TyPtr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`)
    TyRptr(Option<Lifetime>, MutTy),
    /// A bare function (e.g. `fn(usize) -> bool`)
    TyBareFn(P<BareFnTy>),
    /// The never type (`!`)
    TyNever,
    /// A tuple (`(A, B, C, D,...)`)
    TyTup(HirVec<P<Ty>>),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type, e.g. `<Vec<T> as Trait>::Type` or `<T>::Target`.
    ///
    /// Type parameters may be stored in each `PathSegment`.
    TyPath(QPath),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TyTraitObject(TyParamBounds),
    /// An `impl Bound1 + Bound2 + Bound3` type
    /// where `Bound` is a trait or a lifetime.
    TyImplTrait(TyParamBounds),
    /// Unused for now
    TyTypeof(BodyId),
    /// TyInfer means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    TyInfer,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsmOutput {
    pub constraint: Symbol,
    pub is_rw: bool,
    pub is_indirect: bool,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsm {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: HirVec<InlineAsmOutput>,
    pub inputs: HirVec<Symbol>,
    pub clobbers: HirVec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    pub expn_id: ExpnId,
}

/// represents an argument in a function header
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arg {
    pub pat: P<Pat>,
    pub id: NodeId,
}

/// Represents the header (not the body) of a function declaration
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct FnDecl {
    pub inputs: HirVec<P<Ty>>,
    pub output: FunctionRetTy,
    pub variadic: bool,
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
    Default { has_value: bool },
    Final,
}

impl Defaultness {
    pub fn has_value(&self) -> bool {
        match *self {
            Defaultness::Default { has_value, .. } => has_value,
            Defaultness::Final => true,
        }
    }

    pub fn is_final(&self) -> bool {
        *self == Defaultness::Final
    }

    pub fn is_default(&self) -> bool {
        match *self {
            Defaultness::Default { .. } => true,
            _ => false,
        }
    }
}

impl fmt::Display for Unsafety {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(match *self {
                              Unsafety::Normal => "normal",
                              Unsafety::Unsafe => "unsafe",
                          },
                          f)
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
    DefaultReturn(Span),
    /// Everything else
    Return(P<Ty>),
}

impl FunctionRetTy {
    pub fn span(&self) -> Span {
        match *self {
            DefaultReturn(span) => span,
            Return(ref ty) => ty.span,
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: HirVec<ItemId>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignMod {
    pub abi: Abi,
    pub items: HirVec<ForeignItem>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct EnumDef {
    pub variants: HirVec<Variant>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Variant_ {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub data: VariantData,
    /// Explicit discriminant, eg `Foo = 1`
    pub disr_expr: Option<BodyId>,
}

pub type Variant = Spanned<Variant_>;

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum UseKind {
    /// One import, e.g. `use foo::bar` or `use foo::bar as baz`.
    /// Also produced for each element of a list `use`, e.g.
    // `use foo::{a, b}` lowers to `use foo::a; use foo::b;`.
    Single,

    /// Glob import, e.g. `use foo::*`.
    Glob,

    /// Degenerate list import, e.g. `use foo::{a, b}` produces
    /// an additional `use foo::{}` for performing checks such as
    /// unstable feature gating. May be removed in the future.
    ListStem,
}

/// TraitRef's appear in impls.
///
/// resolve maps each TraitRef's ref_id to its defining trait; that's all
/// that the ref_id is for. Note that ref_id's value is not the NodeId of the
/// trait being referred to but just a unique NodeId that serves as a key
/// within the DefMap.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PolyTraitRef {
    /// The `'a` in `<'a> Foo<&'a T>`
    pub bound_lifetimes: HirVec<LifetimeDef>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef,

    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Visibility {
    Public,
    Crate,
    Restricted { path: P<Path>, id: NodeId },
    Inherited,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct StructField {
    pub span: Span,
    pub name: Name,
    pub vis: Visibility,
    pub id: NodeId,
    pub ty: P<Ty>,
    pub attrs: HirVec<Attribute>,
}

impl StructField {
    // Still necessary in couple of places
    pub fn is_positional(&self) -> bool {
        let first = self.name.as_str().as_bytes()[0];
        first >= b'0' && first <= b'9'
    }
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
    Struct(HirVec<StructField>, NodeId),
    Tuple(HirVec<StructField>, NodeId),
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
            VariantData::Struct(_, id) | VariantData::Tuple(_, id) | VariantData::Unit(id) => id,
        }
    }
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_tuple(&self) -> bool {
        if let VariantData::Tuple(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_unit(&self) -> bool {
        if let VariantData::Unit(..) = *self {
            true
        } else {
            false
        }
    }
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ItemId {
    pub id: NodeId,
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Item {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub id: NodeId,
    pub node: Item_,
    pub vis: Visibility,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Item_ {
    /// An`extern crate` item, with optional original crate name,
    ///
    /// e.g. `extern crate foo` or `extern crate foo_bar as foo`
    ItemExternCrate(Option<Name>),

    /// `use foo::bar::*;` or `use foo::bar::baz as quux;`
    ///
    /// or just
    ///
    /// `use foo::bar::baz;` (with `as baz` implicitly on the right)
    ItemUse(P<Path>, UseKind),

    /// A `static` item
    ItemStatic(P<Ty>, Mutability, BodyId),
    /// A `const` item
    ItemConst(P<Ty>, BodyId),
    /// A function declaration
    ItemFn(P<FnDecl>, Unsafety, Constness, Abi, Generics, BodyId),
    /// A module
    ItemMod(Mod),
    /// An external module
    ItemForeignMod(ForeignMod),
    /// A type alias, e.g. `type Foo = Bar<u8>`
    ItemTy(P<Ty>, Generics),
    /// An enum definition, e.g. `enum Foo<A, B> {C<A>, D<B>}`
    ItemEnum(EnumDef, Generics),
    /// A struct definition, e.g. `struct Foo<A> {x: A}`
    ItemStruct(VariantData, Generics),
    /// A union definition, e.g. `union Foo<A, B> {x: A, y: B}`
    ItemUnion(VariantData, Generics),
    /// Represents a Trait Declaration
    ItemTrait(Unsafety, Generics, TyParamBounds, HirVec<TraitItemRef>),

    // Default trait implementations
    ///
    /// `impl Trait for .. {}`
    ItemDefaultImpl(Unsafety, TraitRef),
    /// An implementation, eg `impl<A> Trait for Foo { .. }`
    ItemImpl(Unsafety,
             ImplPolarity,
             Generics,
             Option<TraitRef>, // (optional) trait this impl implements
             P<Ty>, // self
             HirVec<ImplItemRef>),
}

impl Item_ {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ItemExternCrate(..) => "extern crate",
            ItemUse(..) => "use",
            ItemStatic(..) => "static item",
            ItemConst(..) => "constant item",
            ItemFn(..) => "function",
            ItemMod(..) => "module",
            ItemForeignMod(..) => "foreign module",
            ItemTy(..) => "type alias",
            ItemEnum(..) => "enum",
            ItemStruct(..) => "struct",
            ItemUnion(..) => "union",
            ItemTrait(..) => "trait",
            ItemImpl(..) |
            ItemDefaultImpl(..) => "item",
        }
    }
}

/// A reference from an trait to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitItemRef {
    pub id: TraitItemId,
    pub name: Name,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub defaultness: Defaultness,
}

/// A reference from an impl to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ImplItemRef {
    pub id: ImplItemId,
    pub name: Name,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub vis: Visibility,
    pub defaultness: Defaultness,
}

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum AssociatedItemKind {
    Const,
    Method { has_self: bool },
    Type,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignItem {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub node: ForeignItem_,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

/// An item within an `extern` block
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ForeignItem_ {
    /// A foreign function
    ForeignItemFn(P<FnDecl>, HirVec<Spanned<Name>>, Generics),
    /// A foreign static item (`static ext: u8`), with optional mutability
    /// (the boolean is true when mutable)
    ForeignItemStatic(P<Ty>, bool),
}

impl ForeignItem_ {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ForeignItemFn(..) => "foreign function",
            ForeignItemStatic(..) => "foreign static item",
        }
    }
}

/// A free variable referred to in a function.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Freevar {
    /// The variable being accessed free.
    pub def: Def,

    // First span where it is accessed (there can be multiple).
    pub span: Span
}

pub type FreevarMap = NodeMap<Vec<Freevar>>;

pub type CaptureModeMap = NodeMap<CaptureClause>;

#[derive(Clone, Debug)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_id: Option<NodeId>,
}

// Trait method resolution
pub type TraitMap = NodeMap<Vec<TraitCandidate>>;

// Map from the NodeId of a glob import to a list of items which are actually
// imported.
pub type GlobMap = NodeMap<FxHashSet<Name>>;
