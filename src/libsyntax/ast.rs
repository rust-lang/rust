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
pub use self::ViewPath_::*;
pub use self::PathParameters::*;
pub use util::ThinVec;

use syntax_pos::{mk_sp, Span, DUMMY_SP, ExpnId};
use codemap::{respan, Spanned};
use abi::Abi;
use errors;
use parse::token::{self, keywords, InternedString};
use print::pprust;
use ptr::P;
use tokenstream::{TokenTree};

use std::fmt;
use std::rc::Rc;
use std::hash::{Hash, Hasher};
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A name is a part of an identifier, representing a string or gensym. It's
/// the result of interning.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(pub u32);

/// A SyntaxContext represents a chain of macro-expandings
/// and renamings. Each macro expansion corresponds to
/// a fresh u32. This u32 is a reference to a table stored
/// in thread-local storage.
/// The special value EMPTY_CTXT is used to indicate an empty
/// syntax context.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct SyntaxContext(pub u32);

/// An identifier contains a Name (index into the interner
/// table) and a SyntaxContext to track renaming and
/// macro expansion per Flatt et al., "Macros That Work Together"
#[derive(Clone, Copy, Eq)]
pub struct Ident {
    pub name: Name,
    pub ctxt: SyntaxContext
}

impl Name {
    pub fn as_str(self) -> token::InternedString {
        token::InternedString::new_from_name(self)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({})", self, self.0)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.as_str(), f)
    }
}

impl Encodable for Name {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.as_str())
    }
}

impl Decodable for Name {
    fn decode<D: Decoder>(d: &mut D) -> Result<Name, D::Error> {
        Ok(token::intern(&d.read_str()?[..]))
    }
}

pub const EMPTY_CTXT : SyntaxContext = SyntaxContext(0);

impl Ident {
    pub fn new(name: Name, ctxt: SyntaxContext) -> Ident {
        Ident {name: name, ctxt: ctxt}
    }
    pub const fn with_empty_ctxt(name: Name) -> Ident {
        Ident {name: name, ctxt: EMPTY_CTXT}
    }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        if self.ctxt != other.ctxt {
            // There's no one true way to compare Idents. They can be compared
            // non-hygienically `id1.name == id2.name`, hygienically
            // `mtwt::resolve(id1) == mtwt::resolve(id2)`, or even member-wise
            // `(id1.name, id1.ctxt) == (id2.name, id2.ctxt)` depending on the situation.
            // Ideally, PartialEq should not be implemented for Ident at all, but that
            // would be too impractical, because many larger structures (Token, in particular)
            // including Idents as their parts derive PartialEq and use it for non-hygienic
            // comparisons. That's why PartialEq is implemented and defaults to non-hygienic
            // comparison. Hash is implemented too and is consistent with PartialEq, i.e. only
            // the name of Ident is hashed. Still try to avoid comparing idents in your code
            // (especially as keys in hash maps), use one of the three methods listed above
            // explicitly.
            //
            // If you see this panic, then some idents from different contexts were compared
            // non-hygienically. It's likely a bug. Use one of the three comparison methods
            // listed above explicitly.

            panic!("idents with different contexts are compared with operator `==`: \
                {:?}, {:?}.", self, other);
        }

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
        write!(f, "{}#{}", self.name, self.ctxt.0)
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
        Ok(Ident::with_empty_ctxt(Name::decode(d)?))
    }
}

/// A mark represents a unique id associated with a macro expansion
pub type Mrk = u32;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct Lifetime {
    pub id: NodeId,
    pub span: Span,
    pub name: Name
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

/// A lifetime definition, e.g. `'a: 'b+'c+'d`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct LifetimeDef {
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

impl Path {
    // convert a span and an identifier to the corresponding
    // 1-segment path
    pub fn from_ident(s: Span, identifier: Ident) -> Path {
        Path {
            span: s,
            global: false,
            segments: vec!(
                PathSegment {
                    identifier: identifier,
                    parameters: PathParameters::none()
                }
            ),
        }
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of types.
///
/// E.g. `std`, `String` or `Box<T>`
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
    pub fn none() -> PathParameters {
        PathParameters::AngleBracketed(AngleBracketedParameterData {
            lifetimes: Vec::new(),
            types: P::new(),
            bindings: P::new(),
        })
    }

    pub fn is_empty(&self) -> bool {
        match *self {
            PathParameters::AngleBracketed(ref data) => data.is_empty(),

            // Even if the user supplied no types, something like
            // `X()` is equivalent to `X<(),()>`.
            PathParameters::Parenthesized(..) => false,
        }
    }

    pub fn has_lifetimes(&self) -> bool {
        match *self {
            PathParameters::AngleBracketed(ref data) => !data.lifetimes.is_empty(),
            PathParameters::Parenthesized(_) => false,
        }
    }

    pub fn has_types(&self) -> bool {
        match *self {
            PathParameters::AngleBracketed(ref data) => !data.types.is_empty(),
            PathParameters::Parenthesized(..) => true,
        }
    }

    /// Returns the types that the user wrote. Note that these do not necessarily map to the type
    /// parameters in the parenthesized case.
    pub fn types(&self) -> Vec<&P<Ty>> {
        match *self {
            PathParameters::AngleBracketed(ref data) => {
                data.types.iter().collect()
            }
            PathParameters::Parenthesized(ref data) => {
                data.inputs.iter()
                    .chain(data.output.iter())
                    .collect()
            }
        }
    }

    pub fn lifetimes(&self) -> Vec<&Lifetime> {
        match *self {
            PathParameters::AngleBracketed(ref data) => {
                data.lifetimes.iter().collect()
            }
            PathParameters::Parenthesized(_) => {
                Vec::new()
            }
        }
    }

    pub fn bindings(&self) -> Vec<&TypeBinding> {
        match *self {
            PathParameters::AngleBracketed(ref data) => {
                data.bindings.iter().collect()
            }
            PathParameters::Parenthesized(_) => {
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
    pub types: P<[P<Ty>]>,
    /// Bindings (equality constraints) on associated types, if present.
    ///
    /// E.g., `Foo<A=Bar>`.
    pub bindings: P<[TypeBinding]>,
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

pub type CrateNum = u32;

pub type NodeId = u32;

/// Node id used to represent the root of the crate.
pub const CRATE_NODE_ID: NodeId = 0;

/// When parsing and doing expansions, we initially give all AST nodes this AST
/// node value. Then later, in the renumber pass, we renumber them to have
/// small, positive ids.
pub const DUMMY_NODE_ID: NodeId = !0;

pub trait NodeIdAssigner {
    fn next_node_id(&self) -> NodeId;
    fn peek_node_id(&self) -> NodeId;

    fn diagnostic(&self) -> &errors::Handler {
        panic!("this ID assigner cannot emit diagnostics")
    }
}

/// The AST represents all type param bounds as types.
/// typeck::collect::compute_bounds matches these against
/// the "special" built-in traits (see middle::lang_items) and
/// detects Copy, Send and Sync.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyParamBound {
    TraitTyParamBound(PolyTraitRef, TraitBoundModifier),
    RegionTyParamBound(Lifetime)
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

pub type TyParamBounds = P<[TyParamBound]>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct TyParam {
    pub ident: Ident,
    pub id: NodeId,
    pub bounds: TyParamBounds,
    pub default: Option<P<Ty>>,
    pub span: Span
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Generics {
    pub lifetimes: Vec<LifetimeDef>,
    pub ty_params: P<[TyParam]>,
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

impl Default for Generics {
    fn default() ->  Generics {
        Generics {
            lifetimes: Vec::new(),
            ty_params: P::new(),
            where_clause: WhereClause {
                id: DUMMY_NODE_ID,
                predicates: Vec::new(),
            }
        }
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
    /// A type binding, e.g. `for<'c> Foo: Send+Clone+'c`
    BoundPredicate(WhereBoundPredicate),
    /// A lifetime predicate, e.g. `'a: 'b+'c`
    RegionPredicate(WhereRegionPredicate),
    /// An equality predicate (unsupported)
    EqPredicate(WhereEqPredicate),
}

/// A type bound.
///
/// E.g. `for<'c> Foo: Send+Clone+'c`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any lifetimes from a `for` binding
    pub bound_lifetimes: Vec<LifetimeDef>,
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
    pub path: Path,
    pub ty: P<Ty>,
}

/// The set of MetaItems that define the compilation environment of the crate,
/// used to drive conditional compilation
pub type CrateConfig = Vec<P<MetaItem>>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Crate {
    pub module: Mod,
    pub attrs: Vec<Attribute>,
    pub config: CrateConfig,
    pub span: Span,
    pub exported_macros: Vec<MacroDef>,
}

/// A spanned compile-time attribute item.
///
/// E.g. `#[test]`, `#[derive(..)]` or `#[feature = "foo"]`
pub type MetaItem = Spanned<MetaItemKind>;

/// A compile-time attribute item.
///
/// E.g. `#[test]`, `#[derive(..)]` or `#[feature = "foo"]`
#[derive(Clone, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MetaItemKind {
    /// Word meta item.
    ///
    /// E.g. `test` as in `#[test]`
    Word(InternedString),
    /// List meta item.
    ///
    /// E.g. `derive(..)` as in `#[derive(..)]`
    List(InternedString, Vec<P<MetaItem>>),
    /// Name value meta item.
    ///
    /// E.g. `feature = "foo"` as in `#[feature = "foo"]`
    NameValue(InternedString, Lit),
}

// can't be derived because the MetaItemKind::List requires an unordered comparison
impl PartialEq for MetaItemKind {
    fn eq(&self, other: &MetaItemKind) -> bool {
        use self::MetaItemKind::*;
        match *self {
            Word(ref ns) => match *other {
                Word(ref no) => (*ns) == (*no),
                _ => false
            },
            NameValue(ref ns, ref vs) => match *other {
                NameValue(ref no, ref vo) => {
                    (*ns) == (*no) && vs.node == vo.node
                }
                _ => false
            },
            List(ref ns, ref miss) => match *other {
                List(ref no, ref miso) => {
                    ns == no &&
                        miss.iter().all(|mi| miso.iter().any(|x| x.node == mi.node))
                }
                _ => false
            }
        }
    }
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
            PatKind::Box(ref s) | PatKind::Ref(ref s, _) => {
                s.walk(it)
            }
            PatKind::Vec(ref before, ref slice, ref after) => {
                before.iter().all(|p| p.walk(it)) &&
                slice.iter().all(|p| p.walk(it)) &&
                after.iter().all(|p| p.walk(it))
            }
            PatKind::Wild |
            PatKind::Lit(_) |
            PatKind::Range(_, _) |
            PatKind::Ident(_, _, _) |
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
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BindingMode {
    ByRef(Mutability),
    ByValue(Mutability),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum PatKind {
    /// Represents a wildcard pattern (`_`)
    Wild,

    /// A `PatKind::Ident` may either be a new bound variable (`ref mut binding @ OPT_SUBPATTERN`),
    /// or a unit struct/variant pattern, or a const pattern (in the last two cases the third
    /// field must be `None`). Disambiguation cannot be done with parser alone, so it happens
    /// during name resolution.
    Ident(BindingMode, SpannedIdent, Option<P<Pat>>),

    /// A struct or struct variant pattern, e.g. `Variant {x, y, ..}`.
    /// The `bool` is `true` in the presence of a `..`.
    Struct(Path, Vec<Spanned<FieldPat>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    TupleStruct(Path, Vec<P<Pat>>, Option<usize>),

    /// A possibly qualified path pattern.
    /// Unquailfied path patterns `A::B::C` can legally refer to variants, structs, constants
    /// or associated constants. Quailfied path patterns `<A>::B::C`/`<A as Trait>::B::C` can
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
    /// A range pattern, e.g. `1...2`
    Range(P<Expr>, P<Expr>),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Vec(box [a, b], Some(i), box [y, z])`
    Vec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
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

    Semi(P<Expr>),

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

/// An arm of a 'match'.
///
/// E.g. `0...10 => { println!("match!") }` as in
///
/// ```rust,ignore
/// match n {
///     0...10 => { println!("match!") },
///     // ..
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
    pub ident: SpannedIdent,
    pub expr: P<Expr>,
    pub span: Span,
}

pub type SpannedIdent = Spanned<Ident>;

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
    InPlace(P<Expr>, P<Expr>),
    /// An array (`[a, b, c, d]`)
    Vec(Vec<P<Expr>>),
    /// A function call
    ///
    /// The first field resolves to the function itself,
    /// and the second field is the list of arguments
    Call(P<Expr>, Vec<P<Expr>>),
    /// A method call (`x.foo::<Bar, Baz>(a, b, c, d)`)
    ///
    /// The `SpannedIdent` is the identifier for the method name.
    /// The vector of `Ty`s are the ascripted type parameters for the method
    /// (within the angle brackets).
    ///
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    ///
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(foo, [Bar, Baz], [x, a, b, c, d])`.
    MethodCall(SpannedIdent, Vec<P<Ty>>, Vec<P<Expr>>),
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
    IfLet(P<Pat>, P<Expr>, P<Block>, Option<P<Expr>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    While(P<Expr>, P<Block>, Option<SpannedIdent>),
    /// A while-let loop, with an optional label
    ///
    /// `'label: while let pat = expr { block }`
    ///
    /// This is desugared to a combination of `loop` and `match` expressions.
    WhileLet(P<Pat>, P<Expr>, P<Block>, Option<SpannedIdent>),
    /// A for loop, with an optional label
    ///
    /// `'label: for pat in expr { block }`
    ///
    /// This is desugared to a combination of `loop` and `match` expressions.
    ForLoop(P<Pat>, P<Expr>, P<Block>, Option<SpannedIdent>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    Loop(P<Block>, Option<SpannedIdent>),
    /// A `match` block.
    Match(P<Expr>, Vec<Arm>),
    /// A closure (for example, `move |a, b, c| {a + b + c}`)
    ///
    /// The final span is the span of the argument block `|...|`
    Closure(CaptureBy, P<FnDecl>, P<Block>, Span),
    /// A block (`{ ... }`)
    Block(P<Block>),

    /// An assignment (`a = foo()`)
    Assign(P<Expr>, P<Expr>),
    /// An assignment with an operator
    ///
    /// For example, `a += 1`.
    AssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named struct field (`obj.foo`)
    Field(P<Expr>, SpannedIdent),
    /// Access of an unnamed field of a struct or tuple-struct
    ///
    /// For example, `foo.0`.
    TupField(P<Expr>, Spanned<usize>),
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
    /// A `break`, with an optional label to break
    Break(Option<SpannedIdent>),
    /// A `continue`, with an optional label
    Continue(Option<SpannedIdent>),
    /// A `return`, with an optional value to be returned
    Ret(Option<P<Expr>>),

    /// Output of the `asm!()` macro
    InlineAsm(InlineAsm),

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
}

/// The explicit Self type in a "qualified path". The actual
/// path, including the trait and the associated item, is stored
/// separately. `position` represents the index of the associated
/// item qualified with this Self type.
///
/// ```rust,ignore
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
    pub tts: Vec<TokenTree>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum StrStyle {
    /// A regular string, like `"foo"`
    Cooked,
    /// A raw string, like `r##"foo"##`
    ///
    /// The uint is the number of `#` symbols used
    Raw(usize)
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
    Str(InternedString, StrStyle),
    /// A byte string (`b"foo"`)
    ByteStr(Rc<Vec<u8>>),
    /// A byte char (`b'f'`)
    Byte(u8),
    /// A character literal (`'a'`)
    Char(char),
    /// An integer literal (`1`)
    Int(u64, LitIntType),
    /// A float literal (`1f64` or `1E10f64`)
    Float(InternedString, FloatTy),
    /// A float literal without a suffix (`1.0 or 1.0E10`)
    FloatUnsuffixed(InternedString),
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
    pub node: TraitItemKind,
    pub span: Span,
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
    pub node: ImplItemKind,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ImplItemKind {
    Const(P<Ty>, P<Expr>),
    Method(MethodSig, P<Block>),
    Type(P<Ty>),
    Macro(Mac),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub enum IntTy {
    Is,
    I8,
    I16,
    I32,
    I64,
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
            IntTy::Is => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64"
        }
    }

    pub fn val_to_string(&self, val: i64) -> String {
        // cast to a u64 so we can correctly print INT64_MIN. All integral types
        // are parsed as u64, so we wouldn't want to print an extra negative
        // sign.
        format!("{}{}", val as u64, self.ty_to_string())
    }

    pub fn ty_max(&self) -> u64 {
        match *self {
            IntTy::I8 => 0x80,
            IntTy::I16 => 0x8000,
            IntTy::Is | IntTy::I32 => 0x80000000, // FIXME: actually ni about Is
            IntTy::I64 => 0x8000000000000000
        }
    }

    pub fn bit_width(&self) -> Option<usize> {
        Some(match *self {
            IntTy::Is => return None,
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
        })
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub enum UintTy {
    Us,
    U8,
    U16,
    U32,
    U64,
}

impl UintTy {
    pub fn ty_to_string(&self) -> &'static str {
        match *self {
            UintTy::Us => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64"
        }
    }

    pub fn val_to_string(&self, val: u64) -> String {
        format!("{}{}", val, self.ty_to_string())
    }

    pub fn ty_max(&self) -> u64 {
        match *self {
            UintTy::U8 => 0xff,
            UintTy::U16 => 0xffff,
            UintTy::Us | UintTy::U32 => 0xffffffff, // FIXME: actually ni about Us
            UintTy::U64 => 0xffffffffffffffff
        }
    }

    pub fn bit_width(&self) -> Option<usize> {
        Some(match *self {
            UintTy::Us => return None,
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
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

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
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
    pub lifetimes: Vec<LifetimeDef>,
    pub decl: P<FnDecl>
}

/// The different kinds of types recognized by the compiler
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyKind {
    Vec(P<Ty>),
    /// A fixed length array (`[T; n]`)
    FixedLengthVec(P<Ty>, P<Expr>),
    /// A raw pointer (`*const T` or `*mut T`)
    Ptr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`)
    Rptr(Option<Lifetime>, MutTy),
    /// A bare function (e.g. `fn(usize) -> bool`)
    BareFn(P<BareFnTy>),
    /// A tuple (`(A, B, C, D,...)`)
    Tup(Vec<P<Ty>> ),
    /// A path (`module::module::...::Type`), optionally
    /// "qualified", e.g. `<Vec<T> as SomeTrait>::SomeType`.
    ///
    /// Type parameters are stored in the Path itself
    Path(Option<QSelf>, Path),
    /// Something like `A+B`. Note that `B` must always be a path.
    ObjectSum(P<Ty>, TyParamBounds),
    /// A type like `for<'a> Foo<&'a Bar>`
    PolyTraitRef(TyParamBounds),
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
}

/// Inline assembly dialect.
///
/// E.g. `"intel"` as in `asm!("mov eax, 2" : "={eax}"(result) : : : "intel")``
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum AsmDialect {
    Att,
    Intel,
}

/// Inline assembly.
///
/// E.g. `"={eax}"(result)` as in `asm!("mov eax, 2" : "={eax}"(result) : : : "intel")``
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsmOutput {
    pub constraint: InternedString,
    pub expr: P<Expr>,
    pub is_rw: bool,
    pub is_indirect: bool,
}

/// Inline assembly.
///
/// E.g. `asm!("NOP");`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct InlineAsm {
    pub asm: InternedString,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<InlineAsmOutput>,
    pub inputs: Vec<(InternedString, P<Expr>)>,
    pub clobbers: Vec<InternedString>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    pub expn_id: ExpnId,
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
            if ident.node.name == keywords::SelfValue.name() {
                return match self.ty.node {
                    TyKind::ImplicitSelf => Some(respan(self.pat.span, SelfKind::Value(mutbl))),
                    TyKind::Rptr(lt, MutTy{ref ty, mutbl}) if ty.node == TyKind::ImplicitSelf => {
                        Some(respan(self.pat.span, SelfKind::Region(lt, mutbl)))
                    }
                    _ => Some(respan(mk_sp(self.pat.span.lo, self.ty.span.hi),
                                     SelfKind::Explicit(self.ty.clone(), mutbl))),
                }
            }
        }
        None
    }

    pub fn is_self(&self) -> bool {
        if let PatKind::Ident(_, ident, _) = self.pat.node {
            ident.node.name == keywords::SelfValue.name()
        } else {
            false
        }
    }

    pub fn from_self(eself: ExplicitSelf, eself_ident: SpannedIdent) -> Arg {
        let infer_ty = P(Ty {
            id: DUMMY_NODE_ID,
            node: TyKind::ImplicitSelf,
            span: DUMMY_SP,
        });
        let arg = |mutbl, ty, span| Arg {
            pat: P(Pat {
                id: DUMMY_NODE_ID,
                node: PatKind::Ident(BindingMode::ByValue(mutbl), eself_ident, None),
                span: span,
            }),
            ty: ty,
            id: DUMMY_NODE_ID,
        };
        match eself.node {
            SelfKind::Explicit(ty, mutbl) => {
                arg(mutbl, ty, mk_sp(eself.span.lo, eself_ident.span.hi))
            }
            SelfKind::Value(mutbl) => arg(mutbl, infer_ty, eself.span),
            SelfKind::Region(lt, mutbl) => arg(Mutability::Immutable, P(Ty {
                id: DUMMY_NODE_ID,
                node: TyKind::Rptr(lt, MutTy { ty: infer_ty, mutbl: mutbl }),
                span: DUMMY_SP,
            }), eself.span),
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
    /// Functions with return type `!`that always
    /// raise an error or exit (i.e. never return to the caller)
    None(Span),
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
            FunctionRetTy::None(span) => span,
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

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct EnumDef {
    pub variants: Vec<Variant>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Variant_ {
    pub name: Ident,
    pub attrs: Vec<Attribute>,
    pub data: VariantData,
    /// Explicit discriminant, e.g. `Foo = 1`
    pub disr_expr: Option<P<Expr>>,
}

pub type Variant = Spanned<Variant_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum PathListItemKind {
    Ident {
        name: Ident,
        /// renamed in list, e.g. `use foo::{bar as baz};`
        rename: Option<Ident>,
        id: NodeId
    },
    Mod {
        /// renamed in list, e.g. `use foo::{self as baz};`
        rename: Option<Ident>,
        id: NodeId
    }
}

impl PathListItemKind {
    pub fn id(&self) -> NodeId {
        match *self {
            PathListItemKind::Ident { id, .. } | PathListItemKind::Mod { id, .. } => id
        }
    }

    pub fn name(&self) -> Option<Ident> {
        match *self {
            PathListItemKind::Ident { name, .. } => Some(name),
            PathListItemKind::Mod { .. } => None,
        }
    }

    pub fn rename(&self) -> Option<Ident> {
        match *self {
            PathListItemKind::Ident { rename, .. } | PathListItemKind::Mod { rename, .. } => rename
        }
    }
}

pub type PathListItem = Spanned<PathListItemKind>;

pub type ViewPath = Spanned<ViewPath_>;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ViewPath_ {

    /// `foo::bar::baz as quux`
    ///
    /// or just
    ///
    /// `foo::bar::baz` (with `as baz` implicitly on the right)
    ViewPathSimple(Ident, Path),

    /// `foo::bar::*`
    ViewPathGlob(Path),

    /// `foo::bar::{a,b,c}`
    ViewPathList(Path, Vec<PathListItem>)
}

impl ViewPath_ {
    pub fn path(&self) -> &Path {
        match *self {
            ViewPathSimple(_, ref path) |
            ViewPathGlob (ref path) |
            ViewPathList(ref path, _) => path
        }
    }
}

/// Meta-data associated with an item
pub type Attribute = Spanned<Attribute_>;

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

/// Doc-comments are promoted to attributes that have is_sugared_doc = true
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Attribute_ {
    pub id: AttrId,
    pub style: AttrStyle,
    pub value: P<MetaItem>,
    pub is_sugared_doc: bool,
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
    pub bound_lifetimes: Vec<LifetimeDef>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef,

    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Visibility {
    Public,
    Crate(Span),
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

/*
  FIXME (#3300): Should allow items to be anonymous. Right now
  we just use dummy names for anon items.
 */
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
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum ItemKind {
    /// An`extern crate` item, with optional original crate name.
    ///
    /// E.g. `extern crate foo` or `extern crate foo_bar as foo`
    ExternCrate(Option<Name>),
    /// A use declaration (`use` or `pub use`) item.
    ///
    /// E.g. `use foo;`, `use foo::bar;` or `use foo::bar as FooBar;`
    Use(P<ViewPath>),
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
    Fn(P<FnDecl>, Unsafety, Constness, Abi, Generics, P<Block>),
    /// A module declaration (`mod` or `pub mod`).
    ///
    /// E.g. `mod foo;` or `mod foo { .. }`
    Mod(Mod),
    /// An external module (`extern` or `pub extern`).
    ///
    /// E.g. `extern {}` or `extern "C" {}`
    ForeignMod(ForeignMod),
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
    /// A Trait declaration (`trait` or `pub trait`).
    ///
    /// E.g. `trait Foo { .. }` or `trait Foo<T> { .. }`
    Trait(Unsafety, Generics, TyParamBounds, Vec<TraitItem>),
    // Default trait implementation.
    ///
    /// E.g. `impl Trait for .. {}` or `impl<T> Trait<T> for .. {}`
    DefaultImpl(Unsafety, TraitRef),
    /// An implementation.
    ///
    /// E.g. `impl<A> Foo<A> { .. }` or `impl<A> Trait for Foo<A> { .. }`
    Impl(Unsafety,
             ImplPolarity,
             Generics,
             Option<TraitRef>, // (optional) trait this impl implements
             P<Ty>, // self
             Vec<ImplItem>),
    /// A macro invocation (which includes macro definition).
    ///
    /// E.g. `macro_rules! foo { .. }` or `foo!(..)`
    Mac(Mac),
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
            ItemKind::Ty(..) => "type alias",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Trait(..) => "trait",
            ItemKind::Mac(..) |
            ItemKind::Impl(..) |
            ItemKind::DefaultImpl(..) => "item"
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
}

impl ForeignItemKind {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ForeignItemKind::Fn(..) => "foreign function",
            ForeignItemKind::Static(..) => "foreign static item"
        }
    }
}

/// A macro definition, in this crate or imported from another.
///
/// Not parsed directly, but created on macro import or `macro_rules!` expansion.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct MacroDef {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub id: NodeId,
    pub span: Span,
    pub imported_from: Option<Ident>,
    pub export: bool,
    pub use_locally: bool,
    pub allow_internal_unstable: bool,
    pub body: Vec<TokenTree>,
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
