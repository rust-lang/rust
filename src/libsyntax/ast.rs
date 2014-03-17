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
use abi::AbiSet;
use ast_util;
use opt_vec::OptVec;
use parse::token::{InternedString, special_idents, str_to_ident};
use parse::token;

use std::fmt;
use std::fmt::Show;
use std::option::Option;
use std::rc::Rc;
use std::vec_ng::Vec;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A pointer abstraction. FIXME(eddyb) #10676 use Rc<T> in the future.
pub type P<T> = @T;

/// Construct a P<T> from a T value.
pub fn P<T: 'static>(value: T) -> P<T> {
    @value
}

// FIXME #6993: in librustc, uses of "ident" should be replaced
// by just "Name".

// an identifier contains a Name (index into the interner
// table) and a SyntaxContext to track renaming and
// macro expansion per Flatt et al., "Macros
// That Work Together"
#[deriving(Clone, Hash, Ord, TotalEq, TotalOrd, Show)]
pub struct Ident {
    name: Name,
    ctxt: SyntaxContext
}

impl Ident {
    /// Construct an identifier with the given name and an empty context:
    pub fn new(name: Name) -> Ident { Ident {name: name, ctxt: EMPTY_CTXT}}
}

impl Eq for Ident {
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

impl<S: Encoder> Encodable<S> for Ident {
    fn encode(&self, s: &mut S) {
        s.emit_str(token::get_ident(*self).get());
    }
}

impl<D:Decoder> Decodable<D> for Ident {
    fn decode(d: &mut D) -> Ident {
        str_to_ident(d.read_str())
    }
}

/// Function name (not all functions have names)
pub type FnIdent = Option<Ident>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Lifetime {
    id: NodeId,
    span: Span,
    name: Name
}

// a "Path" is essentially Rust's notion of a name;
// for instance: std::cmp::Eq  .  It's represented
// as a sequence of identifiers, along with a bunch
// of supporting information.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Path {
    span: Span,
    /// A `::foo` path, is relative to the crate root rather than current
    /// module (like paths in an import).
    global: bool,
    /// The segments in the path: the things separated by `::`.
    segments: Vec<PathSegment> ,
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    identifier: Ident,
    /// The lifetime parameters for this path segment.
    lifetimes: Vec<Lifetime>,
    /// The type parameters for this path segment, if present.
    types: OptVec<P<Ty>>,
}

pub type CrateNum = u32;

pub type NodeId = u32;

#[deriving(Clone, TotalEq, TotalOrd, Ord, Eq, Encodable, Decodable, Hash, Show)]
pub struct DefId {
    krate: CrateNum,
    node: NodeId,
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
// detects Copy, Send, Send, and Freeze.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum TyParamBound {
    TraitTyParamBound(TraitRef),
    RegionTyParamBound
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct TyParam {
    ident: Ident,
    id: NodeId,
    bounds: OptVec<TyParamBound>,
    default: Option<P<Ty>>
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Generics {
    lifetimes: Vec<Lifetime>,
    ty_params: OptVec<TyParam>,
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

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum MethodProvenance {
    FromTrait(DefId),
    FromImpl(DefId),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Def {
    DefFn(DefId, Purity),
    DefStaticMethod(/* method */ DefId, MethodProvenance, Purity),
    DefSelfTy(/* trait id */ NodeId),
    DefMod(DefId),
    DefForeignMod(DefId),
    DefStatic(DefId, bool /* is_mutbl */),
    DefArg(NodeId, BindingMode),
    DefLocal(NodeId, BindingMode),
    DefVariant(DefId /* enum */, DefId /* variant */, bool /* is_structure */),
    DefTy(DefId),
    DefTrait(DefId),
    DefPrimTy(PrimTy),
    DefTyParam(DefId, uint),
    DefBinding(NodeId, BindingMode),
    DefUse(DefId),
    DefUpvar(NodeId,  // id of closed over var
              @Def,     // closed over def
              NodeId,  // expr node that creates the closure
              NodeId), // id for the block/body of the closure expr

    /// Note that if it's a tuple struct's definition, the node id of the DefId
    /// may either refer to the item definition's id or the StructDef.ctor_id.
    ///
    /// The cases that I have encountered so far are (this is not exhaustive):
    /// - If it's a ty_path referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the item definition's id.
    /// - If it's an ExprPath referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the StructDef.ctor_id.
    DefStruct(DefId),
    DefTyParamBinder(NodeId), /* struct, impl or trait with ty params */
    DefRegion(NodeId),
    DefLabel(NodeId),
    DefMethod(DefId /* method */, Option<DefId> /* trait */),
}

#[deriving(Clone, Eq, Hash, Encodable, Decodable, Show)]
pub enum DefRegion {
    DefStaticRegion,
    DefEarlyBoundRegion(/* index */ uint, /* lifetime decl */ NodeId),
    DefLateBoundRegion(/* binder_id */ NodeId, /* depth */ uint, /* lifetime decl */ NodeId),
    DefFreeRegion(/* block scope */ NodeId, /* lifetime decl */ NodeId),
}

// The set of MetaItems that define the compilation environment of the crate,
// used to drive conditional compilation
pub type CrateConfig = Vec<@MetaItem> ;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Crate {
    module: Mod,
    attrs: Vec<Attribute> ,
    config: CrateConfig,
    span: Span,
}

pub type MetaItem = Spanned<MetaItem_>;

#[deriving(Clone, Encodable, Decodable, Hash)]
pub enum MetaItem_ {
    MetaWord(InternedString),
    MetaList(InternedString, Vec<@MetaItem> ),
    MetaNameValue(InternedString, Lit),
}

// can't be derived because the MetaList requires an unordered comparison
impl Eq for MetaItem_ {
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

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Block {
    view_items: Vec<ViewItem> ,
    stmts: Vec<@Stmt> ,
    expr: Option<@Expr>,
    id: NodeId,
    rules: BlockCheckMode,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Pat {
    id: NodeId,
    node: Pat_,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct FieldPat {
    ident: Ident,
    pat: @Pat,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum BindingMode {
    BindByRef(Mutability),
    BindByValue(Mutability),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
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
    PatIdent(BindingMode, Path, Option<@Pat>),
    PatEnum(Path, Option<Vec<@Pat> >), /* "none" means a * pattern where
                                     * we don't bind the fields to names */
    PatStruct(Path, Vec<FieldPat> , bool),
    PatTup(Vec<@Pat> ),
    PatUniq(@Pat),
    PatRegion(@Pat), // reference pattern
    PatLit(@Expr),
    PatRange(@Expr, @Expr),
    // [a, b, ..i, y, z] is represented as
    // PatVec(~[a, b], Some(i), ~[y, z])
    PatVec(Vec<@Pat> , Option<@Pat>, Vec<@Pat> )
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash, Show)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Sigil {
    BorrowedSigil,
    OwnedSigil,
    ManagedSigil
}

impl fmt::Show for Sigil {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BorrowedSigil => "&".fmt(f),
            OwnedSigil => "~".fmt(f),
            ManagedSigil => "@".fmt(f),
         }
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum ExprVstore {
    ExprVstoreUniq,                 // ~[1,2,3,4]
    ExprVstoreSlice,                // &[1,2,3,4]
    ExprVstoreMutSlice,             // &mut [1,2,3,4]
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
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

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum UnOp {
    UnBox,
    UnUniq,
    UnDeref,
    UnNot,
    UnNeg
}

pub type Stmt = Spanned<Stmt_>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Stmt_ {
    // could be an item or a local (let) binding:
    StmtDecl(@Decl, NodeId),

    // expr without trailing semi-colon (must have unit type):
    StmtExpr(@Expr, NodeId),

    // expr with trailing semi-colon (may have any type):
    StmtSemi(@Expr, NodeId),

    // bool: is there a trailing sem-colon?
    StmtMac(Mac, bool),
}

// FIXME (pending discussion of #1697, #2178...): local should really be
// a refinement on pat.
/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`
#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct Local {
    ty: P<Ty>,
    pat: @Pat,
    init: Option<@Expr>,
    id: NodeId,
    span: Span,
}

pub type Decl = Spanned<Decl_>;

#[deriving(Eq, Encodable, Decodable, Hash)]
pub enum Decl_ {
    // a local (let) binding:
    DeclLocal(@Local),
    // an item binding:
    DeclItem(@Item),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Arm {
    pats: Vec<@Pat> ,
    guard: Option<@Expr>,
    body: @Expr,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Field {
    ident: SpannedIdent,
    expr: @Expr,
    span: Span,
}

pub type SpannedIdent = Spanned<Ident>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Expr {
    id: NodeId,
    node: Expr_,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Expr_ {
    ExprVstore(@Expr, ExprVstore),
    // First expr is the place; second expr is the value.
    ExprBox(@Expr, @Expr),
    ExprVec(Vec<@Expr> , Mutability),
    ExprCall(@Expr, Vec<@Expr> ),
    ExprMethodCall(Ident, Vec<P<Ty>> , Vec<@Expr> ),
    ExprTup(Vec<@Expr> ),
    ExprBinary(BinOp, @Expr, @Expr),
    ExprUnary(UnOp, @Expr),
    ExprLit(@Lit),
    ExprCast(@Expr, P<Ty>),
    ExprIf(@Expr, P<Block>, Option<@Expr>),
    ExprWhile(@Expr, P<Block>),
    // FIXME #6993: change to Option<Name>
    ExprForLoop(@Pat, @Expr, P<Block>, Option<Ident>),
    // Conditionless loop (can be exited with break, cont, or ret)
    // FIXME #6993: change to Option<Name>
    ExprLoop(P<Block>, Option<Ident>),
    ExprMatch(@Expr, Vec<Arm> ),
    ExprFnBlock(P<FnDecl>, P<Block>),
    ExprProc(P<FnDecl>, P<Block>),
    ExprBlock(P<Block>),

    ExprAssign(@Expr, @Expr),
    ExprAssignOp(BinOp, @Expr, @Expr),
    ExprField(@Expr, Ident, Vec<P<Ty>> ),
    ExprIndex(@Expr, @Expr),

    /// Expression that looks like a "name". For example,
    /// `std::vec::from_elem::<uint>` is an ExprPath that's the "name" part
    /// of a function call.
    ExprPath(Path),

    ExprAddrOf(Mutability, @Expr),
    ExprBreak(Option<Ident>),
    ExprAgain(Option<Ident>),
    ExprRet(Option<@Expr>),

    ExprInlineAsm(InlineAsm),

    ExprMac(Mac),

    // A struct literal expression.
    ExprStruct(Path, Vec<Field> , Option<@Expr> /* base */),

    // A vector literal constructed from one repeated element.
    ExprRepeat(@Expr /* element */, @Expr /* count */, Mutability),

    // No-op: used solely so we can pretty-print faithfully
    ExprParen(@Expr)
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
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
#[doc="For macro invocations; parsing is delegated to the macro"]
pub enum TokenTree {
    // a single token
    TTTok(Span, ::parse::token::Token),
    // a delimited sequence (the delimiters appear as the first
    // and last elements of the vector)
    TTDelim(@Vec<TokenTree> ),

    // These only make sense for right-hand-sides of MBE macros:

    // a kleene-style repetition sequence with a span, a TTForest,
    // an optional separator, and a boolean where true indicates
    // zero or more (..), and false indicates one or more (+).
    TTSeq(Span, @Vec<TokenTree> , Option<::parse::token::Token>, bool),

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

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
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
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Mac_ {
    MacInvocTT(Path, Vec<TokenTree> , SyntaxContext),   // new macro-invocation
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum StrStyle {
    CookedStr,
    RawStr(uint)
}

pub type Lit = Spanned<Lit_>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Lit_ {
    LitStr(InternedString, StrStyle),
    LitBinary(Rc<Vec<u8> >),
    LitChar(u32),
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
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct MutTy {
    ty: P<Ty>,
    mutbl: Mutability,
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct TypeField {
    ident: Ident,
    mt: MutTy,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct TypeMethod {
    ident: Ident,
    attrs: Vec<Attribute> ,
    purity: Purity,
    decl: P<FnDecl>,
    generics: Generics,
    explicit_self: ExplicitSelf,
    id: NodeId,
    span: Span,
}

// A trait method is either required (meaning it doesn't have an
// implementation, just a signature) or provided (meaning it has a default
// implementation).
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum TraitMethod {
    Required(TypeMethod),
    Provided(@Method),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum IntTy {
    TyI,
    TyI8,
    TyI16,
    TyI32,
    TyI64,
}

impl fmt::Show for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "{}", ast_util::int_ty_to_str(*self))
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum UintTy {
    TyU,
    TyU8,
    TyU16,
    TyU32,
    TyU64,
}

impl fmt::Show for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "{}", ast_util::uint_ty_to_str(*self))
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum FloatTy {
    TyF32,
    TyF64,
}

impl fmt::Show for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "{}", ast_util::float_ty_to_str(*self))
    }
}

// NB Eq method appears below.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Ty {
    id: NodeId,
    node: Ty_,
    span: Span,
}

// Not represented directly in the AST, referred to by name through a ty_path.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum PrimTy {
    TyInt(IntTy),
    TyUint(UintTy),
    TyFloat(FloatTy),
    TyStr,
    TyBool,
    TyChar
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
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

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct ClosureTy {
    sigil: Sigil,
    region: Option<Lifetime>,
    lifetimes: Vec<Lifetime>,
    purity: Purity,
    onceness: Onceness,
    decl: P<FnDecl>,
    // Optional optvec distinguishes between "fn()" and "fn:()" so we can
    // implement issue #7264. None means "fn()", which means infer a default
    // bound based on pointer sigil during typeck. Some(Empty) means "fn:()",
    // which means use no bounds (e.g., not even Owned on a ~fn()).
    bounds: Option<OptVec<TyParamBound>>,
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct BareFnTy {
    purity: Purity,
    abis: AbiSet,
    lifetimes: Vec<Lifetime>,
    decl: P<FnDecl>
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Ty_ {
    TyNil,
    TyBot, /* bottom type */
    TyBox(P<Ty>),
    TyUniq(P<Ty>),
    TyVec(P<Ty>),
    TyFixedLengthVec(P<Ty>, @Expr),
    TyPtr(MutTy),
    TyRptr(Option<Lifetime>, MutTy),
    TyClosure(@ClosureTy),
    TyBareFn(@BareFnTy),
    TyTup(Vec<P<Ty>> ),
    TyPath(Path, Option<OptVec<TyParamBound>>, NodeId), // for #7264; see above
    TyTypeof(@Expr),
    // TyInfer means the type should be inferred instead of it having been
    // specified. This can appear anywhere in a type.
    TyInfer,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum AsmDialect {
    AsmAtt,
    AsmIntel
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct InlineAsm {
    asm: InternedString,
    asm_str_style: StrStyle,
    clobbers: InternedString,
    inputs: Vec<(InternedString, @Expr)> ,
    outputs: Vec<(InternedString, @Expr)> ,
    volatile: bool,
    alignstack: bool,
    dialect: AsmDialect
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Arg {
    ty: P<Ty>,
    pat: @Pat,
    id: NodeId,
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
            pat: @Pat {
                id: DUMMY_NODE_ID,
                node: PatIdent(BindByValue(mutability), path, None),
                span: span
            },
            id: DUMMY_NODE_ID
        }
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct FnDecl {
    inputs: Vec<Arg> ,
    output: P<Ty>,
    cf: RetStyle,
    variadic: bool
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Purity {
    UnsafeFn, // declared with "unsafe fn"
    ImpureFn, // declared with "fn"
    ExternFn, // declared with "extern fn"
}

impl fmt::Show for Purity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ImpureFn => "impure".fmt(f),
            UnsafeFn => "unsafe".fmt(f),
            ExternFn => "extern".fmt(f),
        }
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum RetStyle {
    NoReturn, // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    Return, // everything else
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum ExplicitSelf_ {
    SelfStatic,                                // no self
    SelfValue,                                 // `self`
    SelfRegion(Option<Lifetime>, Mutability),  // `&'lt self`, `&'lt mut self`
    SelfUniq                                   // `~self`
}

pub type ExplicitSelf = Spanned<ExplicitSelf_>;

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct Method {
    ident: Ident,
    attrs: Vec<Attribute> ,
    generics: Generics,
    explicit_self: ExplicitSelf,
    purity: Purity,
    decl: P<FnDecl>,
    body: P<Block>,
    id: NodeId,
    span: Span,
    vis: Visibility,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Mod {
    view_items: Vec<ViewItem> ,
    items: Vec<@Item> ,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct ForeignMod {
    abis: AbiSet,
    view_items: Vec<ViewItem> ,
    items: Vec<@ForeignItem> ,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct VariantArg {
    ty: P<Ty>,
    id: NodeId,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum VariantKind {
    TupleVariantKind(Vec<VariantArg> ),
    StructVariantKind(@StructDef),
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct EnumDef {
    variants: Vec<P<Variant>> ,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Variant_ {
    name: Ident,
    attrs: Vec<Attribute> ,
    kind: VariantKind,
    id: NodeId,
    disr_expr: Option<@Expr>,
    vis: Visibility,
}

pub type Variant = Spanned<Variant_>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct PathListIdent_ {
    name: Ident,
    id: NodeId,
}

pub type PathListIdent = Spanned<PathListIdent_>;

pub type ViewPath = Spanned<ViewPath_>;

#[deriving(Eq, Encodable, Decodable, Hash)]
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

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct ViewItem {
    node: ViewItem_,
    attrs: Vec<Attribute> ,
    vis: Visibility,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum ViewItem_ {
    // ident: name used to refer to this crate in the code
    // optional (InternedString,StrStyle): if present, this is a location
    // (containing arbitrary characters) from which to fetch the crate sources
    // For example, extern crate whatever = "github.com/mozilla/rust"
    ViewItemExternCrate(Ident, Option<(InternedString,StrStyle)>, NodeId),
    ViewItemUse(Vec<@ViewPath> ),
}

// Meta-data associated with an item
pub type Attribute = Spanned<Attribute_>;

// Distinguishes between Attributes that decorate items and Attributes that
// are contained as statements within items. These two cases need to be
// distinguished for pretty-printing.
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum AttrStyle {
    AttrOuter,
    AttrInner,
}

// doc-comments are promoted to attributes that have is_sugared_doc = true
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Attribute_ {
    style: AttrStyle,
    value: @MetaItem,
    is_sugared_doc: bool,
}

/*
  TraitRef's appear in impls.
  resolve maps each TraitRef's ref_id to its defining trait; that's all
  that the ref_id is for. The impl_id maps to the "self type" of this impl.
  If this impl is an ItemImpl, the impl_id is redundant (it could be the
  same as the impl's node id).
 */
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct TraitRef {
    path: Path,
    ref_id: NodeId,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Visibility {
    Public,
    Private,
    Inherited,
}

impl Visibility {
    pub fn inherit_from(&self, parent_visibility: Visibility) -> Visibility {
        match self {
            &Inherited => parent_visibility,
            &Public | &Private => *self
        }
    }
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct StructField_ {
    kind: StructFieldKind,
    id: NodeId,
    ty: P<Ty>,
    attrs: Vec<Attribute> ,
}

pub type StructField = Spanned<StructField_>;

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum StructFieldKind {
    NamedField(Ident, Visibility),
    UnnamedField // element of a tuple-like struct
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct StructDef {
    fields: Vec<StructField> , /* fields, not including ctor */
    /* ID of the constructor. This is only used for tuple- or enum-like
     * structs. */
    ctor_id: Option<NodeId>
}

/*
  FIXME (#3300): Should allow items to be anonymous. Right now
  we just use dummy names for anon items.
 */
#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Item {
    ident: Ident,
    attrs: Vec<Attribute> ,
    id: NodeId,
    node: Item_,
    vis: Visibility,
    span: Span,
}

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub enum Item_ {
    ItemStatic(P<Ty>, Mutability, @Expr),
    ItemFn(P<FnDecl>, Purity, AbiSet, Generics, P<Block>),
    ItemMod(Mod),
    ItemForeignMod(ForeignMod),
    ItemTy(P<Ty>, Generics),
    ItemEnum(EnumDef, Generics),
    ItemStruct(@StructDef, Generics),
    ItemTrait(Generics, Vec<TraitRef> , Vec<TraitMethod> ),
    ItemImpl(Generics,
             Option<TraitRef>, // (optional) trait this impl implements
             P<Ty>, // self
             Vec<@Method> ),
    // a macro invocation (which includes macro definition)
    ItemMac(Mac),
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub struct ForeignItem {
    ident: Ident,
    attrs: Vec<Attribute> ,
    node: ForeignItem_,
    id: NodeId,
    span: Span,
    vis: Visibility,
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub enum ForeignItem_ {
    ForeignItemFn(P<FnDecl>, Generics),
    ForeignItemStatic(P<Ty>, /* is_mutbl */ bool),
}

// The data we save and restore about an inlined item or method.  This is not
// part of the AST that we parse from a file, but it becomes part of the tree
// that we trans.
#[deriving(Eq, Encodable, Decodable, Hash)]
pub enum InlinedItem {
    IIItem(@Item),
    IIMethod(DefId /* impl id */, bool /* is provided */, @Method),
    IIForeign(@ForeignItem),
}

#[cfg(test)]
mod test {
    use serialize::json;
    use serialize;
    use codemap::*;
    use super::*;

    use std::vec_ng::Vec;

    fn is_freeze<T: Freeze>() {}

    // Assert that the AST remains Freeze (#10693).
    #[test]
    fn ast_is_freeze() {
        is_freeze::<Item>();
    }

    // are ASTs encodable?
    #[test]
    fn check_asts_encodable() {
        let e = Crate {
            module: Mod {view_items: Vec::new(), items: Vec::new()},
            attrs: Vec::new(),
            config: Vec::new(),
            span: Span {
                lo: BytePos(10),
                hi: BytePos(20),
                expn_info: None,
            },
        };
        // doesn't matter which encoder we use....
        let _f = &e as &serialize::Encodable<json::Encoder>;
    }
}
