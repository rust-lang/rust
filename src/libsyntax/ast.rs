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

use codemap::{Span, Spanned, DUMMY_SP};
use abi::Abi;
use ast_util;
use owned_slice::OwnedSlice;
use parse::token::{InternedString, special_idents, str_to_ident};
use parse::token;

use std::fmt;
use std::fmt::Show;
use std::option::Option;
use std::rc::Rc;
use std::gc::{Gc, GC};
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A pointer abstraction. FIXME(eddyb) #10676 use Rc<T> in the future.
pub type P<T> = Gc<T>;

#[allow(non_snake_case_functions)]
/// Construct a P<T> from a T value.
pub fn P<T: 'static>(value: T) -> P<T> {
    box(GC) value
}

// FIXME #6993: in librustc, uses of "ident" should be replaced
// by just "Name".

// an identifier contains a Name (index into the interner
// table) and a SyntaxContext to track renaming and
// macro expansion per Flatt et al., "Macros
// That Work Together"
#[deriving(Clone, Hash, PartialOrd, Eq, Ord, Show)]
pub struct Ident {
    pub name: Name,
    pub ctxt: SyntaxContext
}

impl Ident {
    /// Construct an identifier with the given name and an empty context:
    pub fn new(name: Name) -> Ident { Ident {name: name, ctxt: EMPTY_CTXT}}
}

impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        if self.ctxt == other.ctxt {
            self.name == other.name
        } else {
            // IF YOU SEE ONE OF THESE FAILS: it means that you're comparing
            // idents that have different contexts. You can't fix this without
            // knowing whether the comparison should be hygienic or non-hygienic.
            // if it should be non-hygienic (most things are), just compare the
            // 'name' fields of the idents. Or, even better, replace the idents
            // with Name's.
            //
            // On the other hand, if the comparison does need to be hygienic,
            // one example and its non-hygienic counterpart would be:
            //      syntax::parse::token::mtwt_token_eq
            //      syntax::ext::tt::macro_parser::token_name_eq
            fail!("not allowed to compare these idents: {:?}, {:?}. \
                   Probably related to issue \\#6993", self, other);
        }
    }
    fn ne(&self, other: &Ident) -> bool {
        ! self.eq(other)
    }
}

/// A SyntaxContext represents a chain of macro-expandings
/// and renamings. Each macro expansion corresponds to
/// a fresh uint

// I'm representing this syntax context as an index into
// a table, in order to work around a compiler bug
// that's causing unreleased memory to cause core dumps
// and also perhaps to save some work in destructor checks.
// the special uint '0' will be used to indicate an empty
// syntax context.

// this uint is a reference to a table stored in thread-local
// storage.
pub type SyntaxContext = u32;
pub static EMPTY_CTXT : SyntaxContext = 0;
pub static ILLEGAL_CTXT : SyntaxContext = 1;

/// A name is a part of an identifier, representing a string or gensym. It's
/// the result of interning.
pub type Name = u32;

/// A mark represents a unique id associated with a macro expansion
pub type Mrk = u32;

impl<S: Encoder<E>, E> Encodable<S, E> for Ident {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_str(token::get_ident(*self).get())
    }
}

impl<D:Decoder<E>, E> Decodable<D, E> for Ident {
    fn decode(d: &mut D) -> Result<Ident, E> {
        Ok(str_to_ident(try!(d.read_str()).as_slice()))
    }
}

/// Function name (not all functions have names)
pub type FnIdent = Option<Ident>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Lifetime {
    pub id: NodeId,
    pub span: Span,
    pub name: Name
}

// a "Path" is essentially Rust's notion of a name;
// for instance: std::cmp::PartialEq  .  It's represented
// as a sequence of identifiers, along with a bunch
// of supporting information.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Path {
    pub span: Span,
    /// A `::foo` path, is relative to the crate root rather than current
    /// module (like paths in an import).
    pub global: bool,
    /// The segments in the path: the things separated by `::`.
    pub segments: Vec<PathSegment> ,
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub identifier: Ident,
    /// The lifetime parameters for this path segment.
    pub lifetimes: Vec<Lifetime>,
    /// The type parameters for this path segment, if present.
    pub types: OwnedSlice<P<Ty>>,
}

pub type CrateNum = u32;

pub type NodeId = u32;

#[deriving(Clone, Eq, Ord, PartialOrd, PartialEq, Encodable, Decodable, Hash, Show)]
pub struct DefId {
    pub krate: CrateNum,
    pub node: NodeId,
}

/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub static LOCAL_CRATE: CrateNum = 0;
pub static CRATE_NODE_ID: NodeId = 0;

// When parsing and doing expansions, we initially give all AST nodes this AST
// node value. Then later, in the renumber pass, we renumber them to have
// small, positive ids.
pub static DUMMY_NODE_ID: NodeId = -1;

// The AST represents all type param bounds as types.
// typeck::collect::compute_bounds matches these against
// the "special" built-in traits (see middle::lang_items) and
// detects Copy, Send and Share.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum TyParamBound {
    TraitTyParamBound(TraitRef),
    StaticRegionTyParamBound,
    UnboxedFnTyParamBound(UnboxedFnTy),
    OtherRegionTyParamBound(Span) // FIXME -- just here until work for #5723 lands
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct TyParam {
    pub ident: Ident,
    pub id: NodeId,
    pub sized: Sized,
    pub bounds: OwnedSlice<TyParamBound>,
    pub default: Option<P<Ty>>,
    pub span: Span
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Generics {
    pub lifetimes: Vec<Lifetime>,
    pub ty_params: OwnedSlice<TyParam>,
}

impl Generics {
    pub fn is_parameterized(&self) -> bool {
        self.lifetimes.len() + self.ty_params.len() > 0
    }
    pub fn is_lt_parameterized(&self) -> bool {
        self.lifetimes.len() > 0
    }
    pub fn is_type_parameterized(&self) -> bool {
        self.ty_params.len() > 0
    }
}

// The set of MetaItems that define the compilation environment of the crate,
// used to drive conditional compilation
pub type CrateConfig = Vec<Gc<MetaItem>>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Crate {
    pub module: Mod,
    pub attrs: Vec<Attribute>,
    pub config: CrateConfig,
    pub span: Span,
}

pub type MetaItem = Spanned<MetaItem_>;

#[deriving(Clone, Encodable, Decodable, Eq, Hash)]
pub enum MetaItem_ {
    MetaWord(InternedString),
    MetaList(InternedString, Vec<Gc<MetaItem>>),
    MetaNameValue(InternedString, Lit),
}

// can't be derived because the MetaList requires an unordered comparison
impl PartialEq for MetaItem_ {
    fn eq(&self, other: &MetaItem_) -> bool {
        match *self {
            MetaWord(ref ns) => match *other {
                MetaWord(ref no) => (*ns) == (*no),
                _ => false
            },
            MetaNameValue(ref ns, ref vs) => match *other {
                MetaNameValue(ref no, ref vo) => {
                    (*ns) == (*no) && vs.node == vo.node
                }
                _ => false
            },
            MetaList(ref ns, ref miss) => match *other {
                MetaList(ref no, ref miso) => {
                    ns == no &&
                        miss.iter().all(|mi| miso.iter().any(|x| x.node == mi.node))
                }
                _ => false
            }
        }
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Block {
    pub view_items: Vec<ViewItem>,
    pub stmts: Vec<Gc<Stmt>>,
    pub expr: Option<Gc<Expr>>,
    pub id: NodeId,
    pub rules: BlockCheckMode,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Pat {
    pub id: NodeId,
    pub node: Pat_,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct FieldPat {
    pub ident: Ident,
    pub pat: Gc<Pat>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum BindingMode {
    BindByRef(Mutability),
    BindByValue(Mutability),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Pat_ {
    PatWild,
    PatWildMulti,
    // A PatIdent may either be a new bound variable,
    // or a nullary enum (in which case the second field
    // is None).
    // In the nullary enum case, the parser can't determine
    // which it is. The resolver determines this, and
    // records this pattern's NodeId in an auxiliary
    // set (of "pat_idents that refer to nullary enums")
    PatIdent(BindingMode, Path, Option<Gc<Pat>>),
    PatEnum(Path, Option<Vec<Gc<Pat>>>), /* "none" means a * pattern where
                                     * we don't bind the fields to names */
    PatStruct(Path, Vec<FieldPat>, bool),
    PatTup(Vec<Gc<Pat>>),
    PatBox(Gc<Pat>),
    PatRegion(Gc<Pat>), // reference pattern
    PatLit(Gc<Expr>),
    PatRange(Gc<Expr>, Gc<Expr>),
    // [a, b, ..i, y, z] is represented as
    // PatVec(~[a, b], Some(i), ~[y, z])
    PatVec(Vec<Gc<Pat>>, Option<Gc<Pat>>, Vec<Gc<Pat>>),
    PatMac(Mac),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash, Show)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum ExprVstore {
    ExprVstoreUniq,                 // ~[1,2,3,4]
    ExprVstoreSlice,                // &[1,2,3,4]
    ExprVstoreMutSlice,             // &mut [1,2,3,4]
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum BinOp {
    BiAdd,
    BiSub,
    BiMul,
    BiDiv,
    BiRem,
    BiAnd,
    BiOr,
    BiBitXor,
    BiBitAnd,
    BiBitOr,
    BiShl,
    BiShr,
    BiEq,
    BiLt,
    BiLe,
    BiNe,
    BiGe,
    BiGt,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum UnOp {
    UnBox,
    UnUniq,
    UnDeref,
    UnNot,
    UnNeg
}

pub type Stmt = Spanned<Stmt_>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Stmt_ {
    // could be an item or a local (let) binding:
    StmtDecl(Gc<Decl>, NodeId),

    // expr without trailing semi-colon (must have unit type):
    StmtExpr(Gc<Expr>, NodeId),

    // expr with trailing semi-colon (may have any type):
    StmtSemi(Gc<Expr>, NodeId),

    // bool: is there a trailing sem-colon?
    StmtMac(Mac, bool),
}

/// Where a local declaration came from: either a true `let ... =
/// ...;`, or one desugared from the pattern of a for loop.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum LocalSource {
    LocalLet,
    LocalFor,
}

// FIXME (pending discussion of #1697, #2178...): local should really be
// a refinement on pat.
/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`
#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Local {
    pub ty: P<Ty>,
    pub pat: Gc<Pat>,
    pub init: Option<Gc<Expr>>,
    pub id: NodeId,
    pub span: Span,
    pub source: LocalSource,
}

pub type Decl = Spanned<Decl_>;

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Decl_ {
    // a local (let) binding:
    DeclLocal(Gc<Local>),
    // an item binding:
    DeclItem(Gc<Item>),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Arm {
    pub attrs: Vec<Attribute>,
    pub pats: Vec<Gc<Pat>>,
    pub guard: Option<Gc<Expr>>,
    pub body: Gc<Expr>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Field {
    pub ident: SpannedIdent,
    pub expr: Gc<Expr>,
    pub span: Span,
}

pub type SpannedIdent = Spanned<Ident>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Expr {
    pub id: NodeId,
    pub node: Expr_,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Expr_ {
    ExprVstore(Gc<Expr>, ExprVstore),
    // First expr is the place; second expr is the value.
    ExprBox(Gc<Expr>, Gc<Expr>),
    ExprVec(Vec<Gc<Expr>>),
    ExprCall(Gc<Expr>, Vec<Gc<Expr>>),
    ExprMethodCall(SpannedIdent, Vec<P<Ty>>, Vec<Gc<Expr>>),
    ExprTup(Vec<Gc<Expr>>),
    ExprBinary(BinOp, Gc<Expr>, Gc<Expr>),
    ExprUnary(UnOp, Gc<Expr>),
    ExprLit(Gc<Lit>),
    ExprCast(Gc<Expr>, P<Ty>),
    ExprIf(Gc<Expr>, P<Block>, Option<Gc<Expr>>),
    ExprWhile(Gc<Expr>, P<Block>),
    // FIXME #6993: change to Option<Name>
    ExprForLoop(Gc<Pat>, Gc<Expr>, P<Block>, Option<Ident>),
    // Conditionless loop (can be exited with break, cont, or ret)
    // FIXME #6993: change to Option<Name>
    ExprLoop(P<Block>, Option<Ident>),
    ExprMatch(Gc<Expr>, Vec<Arm>),
    ExprFnBlock(P<FnDecl>, P<Block>),
    ExprProc(P<FnDecl>, P<Block>),
    ExprBlock(P<Block>),

    ExprAssign(Gc<Expr>, Gc<Expr>),
    ExprAssignOp(BinOp, Gc<Expr>, Gc<Expr>),
    ExprField(Gc<Expr>, SpannedIdent, Vec<P<Ty>>),
    ExprIndex(Gc<Expr>, Gc<Expr>),

    /// Expression that looks like a "name". For example,
    /// `std::slice::from_elem::<uint>` is an ExprPath that's the "name" part
    /// of a function call.
    ExprPath(Path),

    ExprAddrOf(Mutability, Gc<Expr>),
    ExprBreak(Option<Ident>),
    ExprAgain(Option<Ident>),
    ExprRet(Option<Gc<Expr>>),

    ExprInlineAsm(InlineAsm),

    ExprMac(Mac),

    // A struct literal expression.
    ExprStruct(Path, Vec<Field> , Option<Gc<Expr>> /* base */),

    // A vector literal constructed from one repeated element.
    ExprRepeat(Gc<Expr> /* element */, Gc<Expr> /* count */),

    // No-op: used solely so we can pretty-print faithfully
    ExprParen(Gc<Expr>)
}

// When the main rust parser encounters a syntax-extension invocation, it
// parses the arguments to the invocation as a token-tree. This is a very
// loose structure, such that all sorts of different AST-fragments can
// be passed to syntax extensions using a uniform type.
//
// If the syntax extension is an MBE macro, it will attempt to match its
// LHS "matchers" against the provided token tree, and if it finds a
// match, will transcribe the RHS token tree, splicing in any captured
// macro_parser::matched_nonterminals into the TTNonterminals it finds.
//
// The RHS of an MBE macro is the only place a TTNonterminal or TTSeq
// makes any real sense. You could write them elsewhere but nothing
// else knows what to do with them, so you'll probably get a syntax
// error.
//
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
#[doc="For macro invocations; parsing is delegated to the macro"]
pub enum TokenTree {
    // a single token
    TTTok(Span, ::parse::token::Token),
    // a delimited sequence (the delimiters appear as the first
    // and last elements of the vector)
    // FIXME(eddyb) #6308 Use Rc<[TokenTree]> after DST.
    TTDelim(Rc<Vec<TokenTree>>),

    // These only make sense for right-hand-sides of MBE macros:

    // a kleene-style repetition sequence with a span, a TTForest,
    // an optional separator, and a boolean where true indicates
    // zero or more (..), and false indicates one or more (+).
    // FIXME(eddyb) #6308 Use Rc<[TokenTree]> after DST.
    TTSeq(Span, Rc<Vec<TokenTree>>, Option<::parse::token::Token>, bool),

    // a syntactic variable that will be filled in by macro expansion.
    TTNonterminal(Span, Ident)
}

//
// Matchers are nodes defined-by and recognized-by the main rust parser and
// language, but they're only ever found inside syntax-extension invocations;
// indeed, the only thing that ever _activates_ the rules in the rust parser
// for parsing a matcher is a matcher looking for the 'matchers' nonterminal
// itself. Matchers represent a small sub-language for pattern-matching
// token-trees, and are thus primarily used by the macro-defining extension
// itself.
//
// MatchTok
// --------
//
//     A matcher that matches a single token, denoted by the token itself. So
//     long as there's no $ involved.
//
//
// MatchSeq
// --------
//
//     A matcher that matches a sequence of sub-matchers, denoted various
//     possible ways:
//
//             $(M)*       zero or more Ms
//             $(M)+       one or more Ms
//             $(M),+      one or more comma-separated Ms
//             $(A B C);*  zero or more semi-separated 'A B C' seqs
//
//
// MatchNonterminal
// -----------------
//
//     A matcher that matches one of a few interesting named rust
//     nonterminals, such as types, expressions, items, or raw token-trees. A
//     black-box matcher on expr, for example, binds an expr to a given ident,
//     and that ident can re-occur as an interpolation in the RHS of a
//     macro-by-example rule. For example:
//
//        $foo:expr   =>     1 + $foo    // interpolate an expr
//        $foo:tt     =>     $foo        // interpolate a token-tree
//        $foo:tt     =>     bar! $foo   // only other valid interpolation
//                                       // is in arg position for another
//                                       // macro
//
// As a final, horrifying aside, note that macro-by-example's input is
// also matched by one of these matchers. Holy self-referential! It is matched
// by a MatchSeq, specifically this one:
//
//                   $( $lhs:matchers => $rhs:tt );+
//
// If you understand that, you have closed to loop and understand the whole
// macro system. Congratulations.
//
pub type Matcher = Spanned<Matcher_>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Matcher_ {
    // match one token
    MatchTok(::parse::token::Token),
    // match repetitions of a sequence: body, separator, zero ok?,
    // lo, hi position-in-match-array used:
    MatchSeq(Vec<Matcher> , Option<::parse::token::Token>, bool, uint, uint),
    // parse a Rust NT: name to bind, name of NT, position in match array:
    MatchNonterminal(Ident, Ident, uint)
}

pub type Mac = Spanned<Mac_>;

// represents a macro invocation. The Path indicates which macro
// is being invoked, and the vector of token-trees contains the source
// of the macro invocation.
// There's only one flavor, now, so this could presumably be simplified.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Mac_ {
    MacInvocTT(Path, Vec<TokenTree> , SyntaxContext),   // new macro-invocation
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum StrStyle {
    CookedStr,
    RawStr(uint)
}

pub type Lit = Spanned<Lit_>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Lit_ {
    LitStr(InternedString, StrStyle),
    LitBinary(Rc<Vec<u8> >),
    LitChar(char),
    LitInt(i64, IntTy),
    LitUint(u64, UintTy),
    LitIntUnsuffixed(i64),
    LitFloat(InternedString, FloatTy),
    LitFloatUnsuffixed(InternedString),
    LitNil,
    LitBool(bool),
}

// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct TypeField {
    pub ident: Ident,
    pub mt: MutTy,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct TypeMethod {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub fn_style: FnStyle,
    pub decl: P<FnDecl>,
    pub generics: Generics,
    pub explicit_self: ExplicitSelf,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

// A trait method is either required (meaning it doesn't have an
// implementation, just a signature) or provided (meaning it has a default
// implementation).
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum TraitMethod {
    Required(TypeMethod),
    Provided(Gc<Method>),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum IntTy {
    TyI,
    TyI8,
    TyI16,
    TyI32,
    TyI64,
}

impl fmt::Show for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}",
               ast_util::int_ty_to_str(*self, None, ast_util::AutoSuffix))
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum UintTy {
    TyU,
    TyU8,
    TyU16,
    TyU32,
    TyU64,
}

impl fmt::Show for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}",
               ast_util::uint_ty_to_str(*self, None, ast_util::AutoSuffix))
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum FloatTy {
    TyF32,
    TyF64,
    TyF128
}

impl fmt::Show for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", ast_util::float_ty_to_str(*self))
    }
}

// NB PartialEq method appears below.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Ty {
    pub id: NodeId,
    pub node: Ty_,
    pub span: Span,
}

// Not represented directly in the AST, referred to by name through a ty_path.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum PrimTy {
    TyInt(IntTy),
    TyUint(UintTy),
    TyFloat(FloatTy),
    TyStr,
    TyBool,
    TyChar
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Onceness {
    Once,
    Many
}

impl fmt::Show for Onceness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Once => "once".fmt(f),
            Many => "many".fmt(f),
        }
    }
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct ClosureTy {
    pub lifetimes: Vec<Lifetime>,
    pub fn_style: FnStyle,
    pub onceness: Onceness,
    pub decl: P<FnDecl>,
    // Optional optvec distinguishes between "fn()" and "fn:()" so we can
    // implement issue #7264. None means "fn()", which means infer a default
    // bound based on pointer sigil during typeck. Some(Empty) means "fn:()",
    // which means use no bounds (e.g., not even Owned on a ~fn()).
    pub bounds: Option<OwnedSlice<TyParamBound>>,
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct BareFnTy {
    pub fn_style: FnStyle,
    pub abi: Abi,
    pub lifetimes: Vec<Lifetime>,
    pub decl: P<FnDecl>
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct UnboxedFnTy {
    pub decl: P<FnDecl>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Ty_ {
    TyNil,
    TyBot, /* bottom type */
    TyBox(P<Ty>),
    TyUniq(P<Ty>),
    TyVec(P<Ty>),
    TyFixedLengthVec(P<Ty>, Gc<Expr>),
    TyPtr(MutTy),
    TyRptr(Option<Lifetime>, MutTy),
    TyClosure(Gc<ClosureTy>, Option<Lifetime>),
    TyProc(Gc<ClosureTy>),
    TyBareFn(Gc<BareFnTy>),
    TyUnboxedFn(Gc<UnboxedFnTy>),
    TyTup(Vec<P<Ty>> ),
    TyPath(Path, Option<OwnedSlice<TyParamBound>>, NodeId), // for #7264; see above
    // No-op; kept solely so that we can pretty-print faithfully
    TyParen(P<Ty>),
    TyTypeof(Gc<Expr>),
    // TyInfer means the type should be inferred instead of it having been
    // specified. This can appear anywhere in a type.
    TyInfer,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum AsmDialect {
    AsmAtt,
    AsmIntel
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct InlineAsm {
    pub asm: InternedString,
    pub asm_str_style: StrStyle,
    pub clobbers: InternedString,
    pub inputs: Vec<(InternedString, Gc<Expr>)>,
    pub outputs: Vec<(InternedString, Gc<Expr>)>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Arg {
    pub ty: P<Ty>,
    pub pat: Gc<Pat>,
    pub id: NodeId,
}

impl Arg {
    pub fn new_self(span: Span, mutability: Mutability) -> Arg {
        let path = ast_util::ident_to_path(span, special_idents::self_);
        Arg {
            // HACK(eddyb) fake type for the self argument.
            ty: P(Ty {
                id: DUMMY_NODE_ID,
                node: TyInfer,
                span: DUMMY_SP,
            }),
            pat: box(GC) Pat {
                id: DUMMY_NODE_ID,
                node: PatIdent(BindByValue(mutability), path, None),
                span: span
            },
            id: DUMMY_NODE_ID
        }
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct FnDecl {
    pub inputs: Vec<Arg>,
    pub output: P<Ty>,
    pub cf: RetStyle,
    pub variadic: bool
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum FnStyle {
    UnsafeFn, // declared with "unsafe fn"
    NormalFn, // declared with "fn"
}

impl fmt::Show for FnStyle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NormalFn => "normal".fmt(f),
            UnsafeFn => "unsafe".fmt(f),
        }
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum RetStyle {
    NoReturn, // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    Return, // everything else
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum ExplicitSelf_ {
    SelfStatic,                                // no self
    SelfValue,                                 // `self`
    SelfRegion(Option<Lifetime>, Mutability),  // `&'lt self`, `&'lt mut self`
    SelfUniq                                   // `~self`
}

pub type ExplicitSelf = Spanned<ExplicitSelf_>;

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Method {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub generics: Generics,
    pub explicit_self: ExplicitSelf,
    pub fn_style: FnStyle,
    pub decl: P<FnDecl>,
    pub body: P<Block>,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub view_items: Vec<ViewItem>,
    pub items: Vec<Gc<Item>>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct ForeignMod {
    pub abi: Abi,
    pub view_items: Vec<ViewItem>,
    pub items: Vec<Gc<ForeignItem>>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct VariantArg {
    pub ty: P<Ty>,
    pub id: NodeId,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum VariantKind {
    TupleVariantKind(Vec<VariantArg>),
    StructVariantKind(Gc<StructDef>),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct EnumDef {
    pub variants: Vec<P<Variant>>,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Variant_ {
    pub name: Ident,
    pub attrs: Vec<Attribute>,
    pub kind: VariantKind,
    pub id: NodeId,
    pub disr_expr: Option<Gc<Expr>>,
    pub vis: Visibility,
}

pub type Variant = Spanned<Variant_>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct PathListIdent_ {
    pub name: Ident,
    pub id: NodeId,
}

pub type PathListIdent = Spanned<PathListIdent_>;

pub type ViewPath = Spanned<ViewPath_>;

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum ViewPath_ {

    // quux = foo::bar::baz
    //
    // or just
    //
    // foo::bar::baz  (with 'baz =' implicitly on the left)
    ViewPathSimple(Ident, Path, NodeId),

    // foo::bar::*
    ViewPathGlob(Path, NodeId),

    // foo::bar::{a,b,c}
    ViewPathList(Path, Vec<PathListIdent> , NodeId)
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct ViewItem {
    pub node: ViewItem_,
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum ViewItem_ {
    // ident: name used to refer to this crate in the code
    // optional (InternedString,StrStyle): if present, this is a location
    // (containing arbitrary characters) from which to fetch the crate sources
    // For example, extern crate whatever = "github.com/rust-lang/rust"
    ViewItemExternCrate(Ident, Option<(InternedString,StrStyle)>, NodeId),
    ViewItemUse(Gc<ViewPath>),
}

// Meta-data associated with an item
pub type Attribute = Spanned<Attribute_>;

// Distinguishes between Attributes that decorate items and Attributes that
// are contained as statements within items. These two cases need to be
// distinguished for pretty-printing.
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum AttrStyle {
    AttrOuter,
    AttrInner,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct AttrId(pub uint);

// doc-comments are promoted to attributes that have is_sugared_doc = true
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Attribute_ {
    pub id: AttrId,
    pub style: AttrStyle,
    pub value: Gc<MetaItem>,
    pub is_sugared_doc: bool,
}

/*
  TraitRef's appear in impls.
  resolve maps each TraitRef's ref_id to its defining trait; that's all
  that the ref_id is for. The impl_id maps to the "self type" of this impl.
  If this impl is an ItemImpl, the impl_id is redundant (it could be the
  same as the impl's node id).
 */
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Visibility {
    Public,
    Inherited,
}

impl Visibility {
    pub fn inherit_from(&self, parent_visibility: Visibility) -> Visibility {
        match self {
            &Inherited => parent_visibility,
            &Public => *self
        }
    }
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Sized {
    DynSize,
    StaticSize,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct StructField_ {
    pub kind: StructFieldKind,
    pub id: NodeId,
    pub ty: P<Ty>,
    pub attrs: Vec<Attribute>,
}

impl StructField_ {
    pub fn ident(&self) -> Option<Ident> {
        match self.kind {
            NamedField(ref ident, _) => Some(ident.clone()),
            UnnamedField(_) => None
        }
    }
}

pub type StructField = Spanned<StructField_>;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum StructFieldKind {
    NamedField(Ident, Visibility),
    UnnamedField(Visibility), // element of a tuple-like struct
}

impl StructFieldKind {
    pub fn is_unnamed(&self) -> bool {
        match *self {
            UnnamedField(..) => true,
            NamedField(..) => false,
        }
    }
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct StructDef {
    pub fields: Vec<StructField>, /* fields, not including ctor */
    /* ID of the constructor. This is only used for tuple- or enum-like
     * structs. */
    pub ctor_id: Option<NodeId>,
    pub super_struct: Option<P<Ty>>, // Super struct, if specified.
    pub is_virtual: bool,            // True iff the struct may be inherited from.
}

/*
  FIXME (#3300): Should allow items to be anonymous. Right now
  we just use dummy names for anon items.
 */
#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct Item {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub id: NodeId,
    pub node: Item_,
    pub vis: Visibility,
    pub span: Span,
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Item_ {
    ItemStatic(P<Ty>, Mutability, Gc<Expr>),
    ItemFn(P<FnDecl>, FnStyle, Abi, Generics, P<Block>),
    ItemMod(Mod),
    ItemForeignMod(ForeignMod),
    ItemTy(P<Ty>, Generics),
    ItemEnum(EnumDef, Generics),
    ItemStruct(Gc<StructDef>, Generics),
    ItemTrait(Generics, Sized, Vec<TraitRef> , Vec<TraitMethod> ),
    ItemImpl(Generics,
             Option<TraitRef>, // (optional) trait this impl implements
             P<Ty>, // self
             Vec<Gc<Method>>),
    // a macro invocation (which includes macro definition)
    ItemMac(Mac),
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct ForeignItem {
    pub ident: Ident,
    pub attrs: Vec<Attribute>,
    pub node: ForeignItem_,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum ForeignItem_ {
    ForeignItemFn(P<FnDecl>, Generics),
    ForeignItemStatic(P<Ty>, /* is_mutbl */ bool),
}

// The data we save and restore about an inlined item or method.  This is not
// part of the AST that we parse from a file, but it becomes part of the tree
// that we trans.
#[deriving(PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum InlinedItem {
    IIItem(Gc<Item>),
    IIMethod(DefId /* impl id */, bool /* is provided */, Gc<Method>),
    IIForeign(Gc<ForeignItem>),
}

#[cfg(test)]
mod test {
    use serialize::json;
    use serialize;
    use codemap::*;
    use super::*;

    // are ASTs encodable?
    #[test]
    fn check_asts_encodable() {
        use std::io;
        let e = Crate {
            module: Mod {
                inner: Span {
                    lo: BytePos(11),
                    hi: BytePos(19),
                    expn_info: None,
                },
                view_items: Vec::new(),
                items: Vec::new(),
            },
            attrs: Vec::new(),
            config: Vec::new(),
            span: Span {
                lo: BytePos(10),
                hi: BytePos(20),
                expn_info: None,
            },
        };
        // doesn't matter which encoder we use....
        let _f = &e as &serialize::Encodable<json::Encoder, io::IoError>;
    }
}
