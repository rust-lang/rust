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
pub use self::ExplicitSelf_::*;
pub use self::Expr_::*;
pub use self::FunctionRetTy::*;
pub use self::ForeignItem_::*;
pub use self::Item_::*;
pub use self::Mutability::*;
pub use self::Pat_::*;
pub use self::PathListItem_::*;
pub use self::PrimTy::*;
pub use self::Stmt_::*;
pub use self::StructFieldKind::*;
pub use self::TraitItem_::*;
pub use self::Ty_::*;
pub use self::TyParamBound::*;
pub use self::UnOp::*;
pub use self::UnsafeSource::*;
pub use self::ViewPath_::*;
pub use self::Visibility::*;
pub use self::PathParameters::*;

use intravisit::Visitor;
use std::collections::BTreeMap;
use syntax::codemap::{self, Span, Spanned, DUMMY_SP, ExpnId};
use syntax::abi::Abi;
use syntax::ast::{Name, NodeId, DUMMY_NODE_ID, TokenTree, AsmDialect};
use syntax::ast::{Attribute, Lit, StrStyle, FloatTy, IntTy, UintTy, CrateConfig};
use syntax::attr::ThinAttributes;
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::InternedString;
use syntax::ptr::P;

use print::pprust;
use util;

use std::fmt;
use std::hash::{Hash, Hasher};
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// Identifier in HIR
#[derive(Clone, Copy, Eq)]
pub struct Ident {
    /// Hygienic name (renamed), should be used by default
    pub name: Name,
    /// Unhygienic name (original, not renamed), needed in few places in name resolution
    pub unhygienic_name: Name,
}

impl Ident {
    pub fn from_name(name: Name) -> Ident {
        Ident { name: name, unhygienic_name: name }
    }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        self.name == other.name
    }
}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.name, f)
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl Encodable for Ident {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.name.encode(s)
    }
}

impl Decodable for Ident {
    fn decode<D: Decoder>(d: &mut D) -> Result<Ident, D::Error> {
        Ok(Ident::from_name(try!(Name::decode(d))))
    }
}

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
               pprust::lifetime_to_string(self))
    }
}

/// A lifetime definition, eg `'a: 'b+'c+'d`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct LifetimeDef {
    pub lifetime: Lifetime,
    pub bounds: Vec<Lifetime>,
}

/// A "Path" is essentially Rust's notion of a name; for instance:
/// std::cmp::PartialEq  .  It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Path {
    pub span: Span,
    /// A `::foo` path, is relative to the crate root rather than current
    /// module (like paths in an import).
    pub global: bool,
    /// The segments in the path: the things separated by `::`.
    pub segments: Vec<PathSegment>,
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

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub identifier: Ident,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`. Note that
    /// this is more than just simple syntactic sugar; the use of
    /// parens affects the region binding rules, so we preserve the
    /// distinction.
    pub parameters: PathParameters,
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
            lifetimes: Vec::new(),
            types: OwnedSlice::empty(),
            bindings: OwnedSlice::empty(),
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
    pub fn types(&self) -> Vec<&P<Ty>> {
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

    pub fn lifetimes(&self) -> Vec<&Lifetime> {
        match *self {
            AngleBracketedParameters(ref data) => {
                data.lifetimes.iter().collect()
            }
            ParenthesizedParameters(_) => {
                Vec::new()
            }
        }
    }

    pub fn bindings(&self) -> Vec<&P<TypeBinding>> {
        match *self {
            AngleBracketedParameters(ref data) => {
                data.bindings.iter().collect()
            }
            ParenthesizedParameters(_) => {
                Vec::new()
            }
        }
    }
}

/// A path like `Foo<'a, T>`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct AngleBracketedParameterData {
    /// The lifetime parameters for this path segment.
    pub lifetimes: Vec<Lifetime>,
    /// The type parameters for this path segment, if present.
    pub types: OwnedSlice<P<Ty>>,
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A=Bar>`.
    pub bindings: OwnedSlice<P<TypeBinding>>,
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
    pub inputs: Vec<P<Ty>>,

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

pub type TyParamBounds = OwnedSlice<TyParamBound>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TyParam {
    pub name: Name,
    pub id: NodeId,
    pub bounds: TyParamBounds,
    pub default: Option<P<Ty>>,
    pub span: Span,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Generics {
    pub lifetimes: Vec<LifetimeDef>,
    pub ty_params: OwnedSlice<TyParam>,
    pub where_clause: WhereClause,
}

impl Generics {
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

/// A `where` clause in a definition
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereClause {
    pub id: NodeId,
    pub predicates: Vec<WherePredicate>,
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
    pub bound_lifetimes: Vec<LifetimeDef>,
    /// The type being bounded
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (`Clone+Send+'static`)
    pub bounds: OwnedSlice<TyParamBound>,
}

/// A lifetime predicate, e.g. `'a: 'b+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: Vec<Lifetime>,
}

/// An equality predicate (unsupported), e.g. `T=int`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereEqPredicate {
    pub id: NodeId,
    pub span: Span,
    pub path: Path,
    pub ty: P<Ty>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
pub struct Crate {
    pub module: Mod,
    pub attrs: Vec<Attribute>,
    pub config: CrateConfig,
    pub span: Span,
    pub exported_macros: Vec<MacroDef>,

    // NB: We use a BTreeMap here so that `visit_all_items` iterates
    // over the ids in increasing order. In principle it should not
    // matter what order we visit things in, but in *practice* it
    // does, because it can affect the order in which errors are
    // detected, which in turn can make compile-fail tests yield
    // slightly different results.
    pub items: BTreeMap<NodeId, Item>,
}

impl Crate {
    pub fn item(&self, id: NodeId) -> &Item {
        &self.items[&id]
    }

    /// Visits all items in the crate in some determinstic (but
    /// unspecified) order. If you just need to process every item,
    /// but don't care about nesting, this method is the best choice.
    ///
    /// If you do care about nesting -- usually because your algorithm
    /// follows lexical scoping rules -- then you want a different
    /// approach. You should override `visit_nested_item` in your
    /// visitor and then call `intravisit::walk_crate` instead.
    pub fn visit_all_items<'hir, V:Visitor<'hir>>(&'hir self, visitor: &mut V) {
        for (_, item) in &self.items {
            visitor.visit_item(item);
        }
    }
}

/// A macro definition, in this crate or imported from another.
///
/// Not parsed directly, but created on macro import or `macro_rules!` expansion.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MacroDef {
    pub name: Name,
    pub attrs: Vec<Attribute>,
    pub id: NodeId,
    pub span: Span,
    pub imported_from: Option<Name>,
    pub export: bool,
    pub use_locally: bool,
    pub allow_internal_unstable: bool,
    pub body: Vec<TokenTree>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Block {
    /// Statements in a block
    pub stmts: Vec<P<Stmt>>,
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
    pub node: Pat_,
    pub span: Span,
}

impl fmt::Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pat({}: {})", self.id, pprust::pat_to_string(self))
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
pub enum Pat_ {
    /// Represents a wildcard pattern (`_`)
    PatWild,

    /// A PatIdent may either be a new bound variable,
    /// or a nullary enum (in which case the third field
    /// is None).
    ///
    /// In the nullary enum case, the parser can't determine
    /// which it is. The resolver determines this, and
    /// records this pattern's NodeId in an auxiliary
    /// set (of "PatIdents that refer to nullary enums")
    PatIdent(BindingMode, Spanned<Ident>, Option<P<Pat>>),

    /// "None" means a `Variant(..)` pattern where we don't bind the fields to names.
    PatEnum(Path, Option<Vec<P<Pat>>>),

    /// An associated const named using the qualified path `<T>::CONST` or
    /// `<T as Trait>::CONST`. Associated consts from inherent impls can be
    /// referred to as simply `T::CONST`, in which case they will end up as
    /// PatEnum, and the resolver will have to sort that out.
    PatQPath(QSelf, Path),

    /// Destructuring of a struct, e.g. `Foo {x, y, ..}`
    /// The `bool` is `true` in the presence of a `..`
    PatStruct(Path, Vec<Spanned<FieldPat>>, bool),
    /// A tuple pattern `(a, b)`
    PatTup(Vec<P<Pat>>),
    /// A `box` pattern
    PatBox(P<Pat>),
    /// A reference pattern, e.g. `&mut (a, b)`
    PatRegion(P<Pat>, Mutability),
    /// A literal
    PatLit(P<Expr>),
    /// A range pattern, e.g. `1...2`
    PatRange(P<Expr>, P<Expr>),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatVec(box [a, b], Some(i), box [y, z])`
    PatVec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
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

/// A statement
pub type Stmt = Spanned<Stmt_>;

impl fmt::Debug for Stmt_ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Sadness.
        let spanned = codemap::dummy_spanned(self.clone());
        write!(f,
               "stmt({}: {})",
               util::stmt_id(&spanned),
               pprust::stmt_to_string(&spanned))
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
    pub attrs: ThinAttributes,
}

pub type Decl = Spanned<Decl_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Decl_ {
    /// A local (let) binding:
    DeclLocal(P<Local>),
    /// An item binding:
    DeclItem(ItemId),
}

/// represents one arm of a 'match'
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arm {
    pub attrs: Vec<Attribute>,
    pub pats: Vec<P<Pat>>,
    pub guard: Option<P<Expr>>,
    pub body: P<Expr>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Field {
    pub name: Spanned<Name>,
    pub expr: P<Expr>,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
    PushUnsafeBlock(UnsafeSource),
    PopUnsafeBlock(UnsafeSource),
    // Within this block (but outside a PopUnstableBlock), we suspend checking of stability.
    PushUnstableBlock,
    PopUnstableBlock,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

/// An expression
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Expr {
    pub id: NodeId,
    pub node: Expr_,
    pub span: Span,
    pub attrs: ThinAttributes,
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Expr_ {
    /// A `box x` expression.
    ExprBox(P<Expr>),
    /// An array (`[a, b, c, d]`)
    ExprVec(Vec<P<Expr>>),
    /// A function call
    ///
    /// The first field resolves to the function itself,
    /// and the second field is the list of arguments
    ExprCall(P<Expr>, Vec<P<Expr>>),
    /// A method call (`x.foo::<Bar, Baz>(a, b, c, d)`)
    ///
    /// The `Spanned<Name>` is the identifier for the method name.
    /// The vector of `Ty`s are the ascripted type parameters for the method
    /// (within the angle brackets).
    ///
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    ///
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprMethodCall(foo, [Bar, Baz], [x, a, b, c, d])`.
    ExprMethodCall(Spanned<Name>, Vec<P<Ty>>, Vec<P<Expr>>),
    /// A tuple (`(a, b, c ,d)`)
    ExprTup(Vec<P<Expr>>),
    /// A binary operation (For example: `a + b`, `a * b`)
    ExprBinary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (For example: `!x`, `*x`)
    ExprUnary(UnOp, P<Expr>),
    /// A literal (For example: `1u8`, `"foo"`)
    ExprLit(P<Lit>),
    /// A cast (`foo as f64`)
    ExprCast(P<Expr>, P<Ty>),
    /// An `if` block, with an optional else block
    ///
    /// `if expr { block } else { expr }`
    ExprIf(P<Expr>, P<Block>, Option<P<Expr>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    ExprWhile(P<Expr>, P<Block>, Option<Ident>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    ExprLoop(P<Block>, Option<Ident>),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    ExprMatch(P<Expr>, Vec<Arm>, MatchSource),
    /// A closure (for example, `move |a, b, c| {a + b + c}`)
    ExprClosure(CaptureClause, P<FnDecl>, P<Block>),
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
    /// A range (`1..2`, `1..`, or `..2`)
    ExprRange(Option<P<Expr>>, Option<P<Expr>>),

    /// Variable reference, possibly containing `::` and/or type
    /// parameters, e.g. foo::bar::<baz>.
    ///
    /// Optionally "qualified",
    /// e.g. `<Vec<T> as SomeTrait>::SomeType`.
    ExprPath(Option<QSelf>, Path),

    /// A referencing operation (`&a` or `&mut a`)
    ExprAddrOf(Mutability, P<Expr>),
    /// A `break`, with an optional label to break
    ExprBreak(Option<Spanned<Ident>>),
    /// A `continue`, with an optional label
    ExprAgain(Option<Spanned<Ident>>),
    /// A `return`, with an optional value to be returned
    ExprRet(Option<P<Expr>>),

    /// Output of the `asm!()` macro
    ExprInlineAsm(InlineAsm),

    /// A struct literal expression.
    ///
    /// For example, `Foo {x: 1, y: 2}`, or
    /// `Foo {x: 1, .. base}`, where `base` is the `Option<Expr>`.
    ExprStruct(Path, Vec<Field>, Option<P<Expr>>),

    /// A vector literal constructed from one repeated element.
    ///
    /// For example, `[1u8; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    ExprRepeat(P<Expr>, P<Expr>),
}

/// The explicit Self type in a "qualified path". The actual
/// path, including the trait and the associated item, is stored
/// separately. `position` represents the index of the associated
/// item qualified with this Self type.
///
///     <Vec<T> as a::b::Trait>::AssociatedItem
///      ^~~~~     ~~~~~~~~~~~~~~^
///      ty        position = 3
///
///     <Vec<T>>::AssociatedItem
///      ^~~~~    ^
///      ty       position = 0
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct QSelf {
    pub ty: P<Ty>,
    pub position: usize,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum MatchSource {
    Normal,
    IfLetDesugar {
        contains_else_clause: bool,
    },
    WhileLetDesugar,
    ForLoopDesugar,
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

/// Represents a method's signature in a trait declaration,
/// or in an implementation.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MethodSig {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub abi: Abi,
    pub decl: P<FnDecl>,
    pub generics: Generics,
    pub explicit_self: ExplicitSelf,
}

/// Represents a method declaration in a trait declaration, possibly including
/// a default implementation A trait method is either required (meaning it
/// doesn't have an implementation, just a signature) or provided (meaning it
/// has a default implementation).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitItem {
    pub id: NodeId,
    pub name: Name,
    pub attrs: Vec<Attribute>,
    pub node: TraitItem_,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitItem_ {
    ConstTraitItem(P<Ty>, Option<P<Expr>>),
    MethodTraitItem(MethodSig, Option<P<Block>>),
    TypeTraitItem(TyParamBounds, Option<P<Ty>>),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ImplItem {
    pub id: NodeId,
    pub name: Name,
    pub vis: Visibility,
    pub attrs: Vec<Attribute>,
    pub node: ImplItemKind,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ImplItemKind {
    Const(P<Ty>, P<Expr>),
    Method(MethodSig, P<Block>),
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


// NB PartialEq method appears below.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub struct Ty {
    pub id: NodeId,
    pub node: Ty_,
    pub span: Span,
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "type({})", pprust::ty_to_string(self))
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
    pub lifetimes: Vec<LifetimeDef>,
    pub decl: P<FnDecl>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
/// The different kinds of types recognized by the compiler
pub enum Ty_ {
    TyVec(P<Ty>),
    /// A fixed length array (`[T; n]`)
    TyFixedLengthVec(P<Ty>, P<Expr>),
    /// A raw pointer (`*const T` or `*mut T`)
    TyPtr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`)
    TyRptr(Option<Lifetime>, MutTy),
    /// A bare function (e.g. `fn(usize) -> bool`)
    TyBareFn(P<BareFnTy>),
    /// A tuple (`(A, B, C, D,...)`)
    TyTup(Vec<P<Ty>>),
    /// A path (`module::module::...::Type`), optionally
    /// "qualified", e.g. `<Vec<T> as SomeTrait>::SomeType`.
    ///
    /// Type parameters are stored in the Path itself
    TyPath(Option<QSelf>, Path),
    /// Something like `A+B`. Note that `B` must always be a path.
    TyObjectSum(P<Ty>, TyParamBounds),
    /// A type like `for<'a> Foo<&'a Bar>`
    TyPolyTraitRef(TyParamBounds),
    /// Unused for now
    TyTypeof(P<Expr>),
    /// TyInfer means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    TyInfer,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsm {
    pub asm: InternedString,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<(InternedString, P<Expr>, bool)>,
    pub inputs: Vec<(InternedString, P<Expr>)>,
    pub clobbers: Vec<InternedString>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    pub expn_id: ExpnId,
}

/// represents an argument in a function header
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Arg {
    pub ty: P<Ty>,
    pub pat: P<Pat>,
    pub id: NodeId,
}

impl Arg {
    pub fn new_self(span: Span, mutability: Mutability, self_ident: Ident) -> Arg {
        let path = Spanned {
            span: span,
            node: self_ident,
        };
        Arg {
            // HACK(eddyb) fake type for the self argument.
            ty: P(Ty {
                id: DUMMY_NODE_ID,
                node: TyInfer,
                span: DUMMY_SP,
            }),
            pat: P(Pat {
                id: DUMMY_NODE_ID,
                node: PatIdent(BindByValue(mutability), path, None),
                span: span,
            }),
            id: DUMMY_NODE_ID,
        }
    }
}

/// Represents the header (not the body) of a function declaration
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct FnDecl {
    pub inputs: Vec<Arg>,
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
    /// Functions with return type `!`that always
    /// raise an error or exit (i.e. never return to the caller)
    NoReturn(Span),
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
            NoReturn(span) => span,
            DefaultReturn(span) => span,
            Return(ref ty) => ty.span,
        }
    }
}

/// Represents the kind of 'self' associated with a method
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ExplicitSelf_ {
    /// No self
    SelfStatic,
    /// `self`
    SelfValue(Name),
    /// `&'lt self`, `&'lt mut self`
    SelfRegion(Option<Lifetime>, Mutability, Name),
    /// `self: TYPE`
    SelfExplicit(P<Ty>, Name),
}

pub type ExplicitSelf = Spanned<ExplicitSelf_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: Vec<ItemId>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignMod {
    pub abi: Abi,
    pub items: Vec<P<ForeignItem>>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct EnumDef {
    pub variants: Vec<P<Variant>>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Variant_ {
    pub name: Name,
    pub attrs: Vec<Attribute>,
    pub data: VariantData,
    /// Explicit discriminant, eg `Foo = 1`
    pub disr_expr: Option<P<Expr>>,
}

pub type Variant = Spanned<Variant_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum PathListItem_ {
    PathListIdent {
        name: Name,
        /// renamed in list, eg `use foo::{bar as baz};`
        rename: Option<Name>,
        id: NodeId,
    },
    PathListMod {
        /// renamed in list, eg `use foo::{self as baz};`
        rename: Option<Name>,
        id: NodeId,
    },
}

impl PathListItem_ {
    pub fn id(&self) -> NodeId {
        match *self {
            PathListIdent { id, .. } | PathListMod { id, .. } => id,
        }
    }

    pub fn name(&self) -> Option<Name> {
        match *self {
            PathListIdent { name, .. } => Some(name),
            PathListMod { .. } => None,
        }
    }

    pub fn rename(&self) -> Option<Name> {
        match *self {
            PathListIdent { rename, .. } | PathListMod { rename, .. } => rename,
        }
    }
}

pub type PathListItem = Spanned<PathListItem_>;

pub type ViewPath = Spanned<ViewPath_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ViewPath_ {
    /// `foo::bar::baz as quux`
    ///
    /// or just
    ///
    /// `foo::bar::baz` (with `as baz` implicitly on the right)
    ViewPathSimple(Name, Path),

    /// `foo::bar::*`
    ViewPathGlob(Path),

    /// `foo::bar::{a,b,c}`
    ViewPathList(Path, Vec<PathListItem>),
}

/// TraitRef's appear in impls.
///
/// resolve maps each TraitRef's ref_id to its defining trait; that's all
/// that the ref_id is for. The impl_id maps to the "self type" of this impl.
/// If this impl is an ItemImpl, the impl_id is redundant (it could be the
/// same as the impl's node id).
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct PolyTraitRef {
    /// The `'a` in `<'a> Foo<&'a T>`
    pub bound_lifetimes: Vec<LifetimeDef>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef,

    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Visibility {
    Public,
    Inherited,
}

impl Visibility {
    pub fn inherit_from(&self, parent_visibility: Visibility) -> Visibility {
        match self {
            &Inherited => parent_visibility,
            &Public => *self,
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct StructField_ {
    pub kind: StructFieldKind,
    pub id: NodeId,
    pub ty: P<Ty>,
    pub attrs: Vec<Attribute>,
}

impl StructField_ {
    pub fn name(&self) -> Option<Name> {
        match self.kind {
            NamedField(name, _) => Some(name),
            UnnamedField(_) => None,
        }
    }
}

pub type StructField = Spanned<StructField_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum StructFieldKind {
    NamedField(Name, Visibility),
    /// Element of a tuple-like struct
    UnnamedField(Visibility),
}

impl StructFieldKind {
    pub fn is_unnamed(&self) -> bool {
        match *self {
            UnnamedField(..) => true,
            NamedField(..) => false,
        }
    }

    pub fn visibility(&self) -> Visibility {
        match *self {
            NamedField(_, vis) | UnnamedField(vis) => vis,
        }
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
    Struct(Vec<StructField>, NodeId),
    Tuple(Vec<StructField>, NodeId),
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

//  FIXME (#3300): Should allow items to be anonymous. Right now
//  we just use dummy names for anon items.
/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Item {
    pub name: Name,
    pub attrs: Vec<Attribute>,
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
    /// A `use` or `pub use` item
    ItemUse(P<ViewPath>),

    /// A `static` item
    ItemStatic(P<Ty>, Mutability, P<Expr>),
    /// A `const` item
    ItemConst(P<Ty>, P<Expr>),
    /// A function declaration
    ItemFn(P<FnDecl>, Unsafety, Constness, Abi, Generics, P<Block>),
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
    /// Represents a Trait Declaration
    ItemTrait(Unsafety, Generics, TyParamBounds, Vec<P<TraitItem>>),

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
             Vec<P<ImplItem>>),
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
            ItemTrait(..) => "trait",
            ItemImpl(..) |
            ItemDefaultImpl(..) => "item",
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ForeignItem {
    pub name: Name,
    pub attrs: Vec<Attribute>,
    pub node: ForeignItem_,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

/// An item within an `extern` block
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ForeignItem_ {
    /// A foreign function
    ForeignItemFn(P<FnDecl>, Generics),
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
