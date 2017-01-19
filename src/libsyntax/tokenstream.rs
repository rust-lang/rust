// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Token Streams
//!
//! TokenStreams represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves either a single Token, a Delimited subsequence of tokens,
//! or a SequenceRepetition specifier (for the purpose of sequence generation during macro
//! expansion).
//!
//! ## Ownership
//! TokenStreams are persistant data structures construced as ropes with reference
//! counted-children. In general, this means that calling an operation on a TokenStream
//! (such as `slice`) produces an entirely new TokenStream from the borrowed reference to
//! the original. This essentially coerces TokenStreams into 'views' of their subparts,
//! and a borrowed TokenStream is sufficient to build an owned TokenStream without taking
//! ownership of the original.

use ast::{self, AttrStyle, LitKind};
use syntax_pos::{Span, DUMMY_SP, NO_EXPANSION};
use codemap::{Spanned, combine_spans};
use ext::base;
use ext::tt::macro_parser;
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::{self, Directory};
use parse::token::{self, Token, Lit, Nonterminal};
use print::pprust;
use symbol::Symbol;

use std::fmt;
use std::iter::*;
use std::ops::{self, Index};
use std::rc::Rc;

/// A delimited sequence of token trees
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Delimited {
    /// The type of delimiter
    pub delim: token::DelimToken,
    /// The span covering the opening delimiter
    pub open_span: Span,
    /// The delimited sequence of token trees
    pub tts: Vec<TokenTree>,
    /// The span covering the closing delimiter
    pub close_span: Span,
}

impl Delimited {
    /// Returns the opening delimiter as a token.
    pub fn open_token(&self) -> token::Token {
        token::OpenDelim(self.delim)
    }

    /// Returns the closing delimiter as a token.
    pub fn close_token(&self) -> token::Token {
        token::CloseDelim(self.delim)
    }

    /// Returns the opening delimiter as a token tree.
    pub fn open_tt(&self) -> TokenTree {
        TokenTree::Token(self.open_span, self.open_token())
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(&self) -> TokenTree {
        TokenTree::Token(self.close_span, self.close_token())
    }

    /// Returns the token trees inside the delimiters.
    pub fn subtrees(&self) -> &[TokenTree] {
        &self.tts
    }
}

/// A sequence of token trees
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct SequenceRepetition {
    /// The sequence of token trees
    pub tts: Vec<TokenTree>,
    /// The optional separator
    pub separator: Option<token::Token>,
    /// Whether the sequence can be repeated zero (*), or one or more times (+)
    pub op: KleeneOp,
    /// The number of `MatchNt`s that appear in the sequence (and subsequences)
    pub num_captures: usize,
}

/// A Kleene-style [repetition operator](http://en.wikipedia.org/wiki/Kleene_star)
/// for token sequences.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

/// When the main rust parser encounters a syntax-extension invocation, it
/// parses the arguments to the invocation as a token-tree. This is a very
/// loose structure, such that all sorts of different AST-fragments can
/// be passed to syntax extensions using a uniform type.
///
/// If the syntax extension is an MBE macro, it will attempt to match its
/// LHS token tree against the provided token tree, and if it finds a
/// match, will transcribe the RHS token tree, splicing in any captured
/// macro_parser::matched_nonterminals into the `SubstNt`s it finds.
///
/// The RHS of an MBE macro is the only place `SubstNt`s are substituted.
/// Nothing special happens to misnamed or misplaced `SubstNt`s.
#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum TokenTree {
    /// A single token
    Token(Span, token::Token),
    /// A delimited sequence of token trees
    Delimited(Span, Rc<Delimited>),

    // This only makes sense in MBE macros.
    /// A kleene-style repetition sequence with a span
    Sequence(Span, Rc<SequenceRepetition>),
}

impl TokenTree {
    pub fn len(&self) -> usize {
        match *self {
            TokenTree::Token(_, token::DocComment(name)) => {
                match doc_comment_style(&name.as_str()) {
                    AttrStyle::Outer => 2,
                    AttrStyle::Inner => 3,
                }
            }
            TokenTree::Token(_, token::Interpolated(ref nt)) => {
                if let Nonterminal::NtTT(..) = **nt { 1 } else { 0 }
            },
            TokenTree::Token(_, token::MatchNt(..)) => 3,
            TokenTree::Delimited(_, ref delimed) => match delimed.delim {
                token::NoDelim => delimed.tts.len(),
                _ => delimed.tts.len() + 2,
            },
            TokenTree::Sequence(_, ref seq) => seq.tts.len(),
            TokenTree::Token(..) => 0,
        }
    }

    pub fn get_tt(&self, index: usize) -> TokenTree {
        match (self, index) {
            (&TokenTree::Token(sp, token::DocComment(_)), 0) => TokenTree::Token(sp, token::Pound),
            (&TokenTree::Token(sp, token::DocComment(name)), 1)
                if doc_comment_style(&name.as_str()) == AttrStyle::Inner => {
                TokenTree::Token(sp, token::Not)
            }
            (&TokenTree::Token(sp, token::DocComment(name)), _) => {
                let stripped = strip_doc_comment_decoration(&name.as_str());

                // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
                // required to wrap the text.
                let num_of_hashes = stripped.chars()
                    .scan(0, |cnt, x| {
                        *cnt = if x == '"' {
                            1
                        } else if *cnt != 0 && x == '#' {
                            *cnt + 1
                        } else {
                            0
                        };
                        Some(*cnt)
                    })
                    .max()
                    .unwrap_or(0);

                TokenTree::Delimited(sp, Rc::new(Delimited {
                    delim: token::Bracket,
                    open_span: sp,
                    tts: vec![TokenTree::Token(sp, token::Ident(ast::Ident::from_str("doc"))),
                              TokenTree::Token(sp, token::Eq),
                              TokenTree::Token(sp, token::Literal(
                                  token::StrRaw(Symbol::intern(&stripped), num_of_hashes), None))],
                    close_span: sp,
                }))
            }
            (&TokenTree::Delimited(_, ref delimed), _) if delimed.delim == token::NoDelim => {
                delimed.tts[index].clone()
            }
            (&TokenTree::Delimited(_, ref delimed), _) => {
                if index == 0 {
                    return delimed.open_tt();
                }
                if index == delimed.tts.len() + 1 {
                    return delimed.close_tt();
                }
                delimed.tts[index - 1].clone()
            }
            (&TokenTree::Token(sp, token::MatchNt(name, kind)), _) => {
                let v = [TokenTree::Token(sp, token::SubstNt(name)),
                         TokenTree::Token(sp, token::Colon),
                         TokenTree::Token(sp, token::Ident(kind))];
                v[index].clone()
            }
            (&TokenTree::Sequence(_, ref seq), _) => seq.tts[index].clone(),
            _ => panic!("Cannot expand a token tree"),
        }
    }

    /// Returns the `Span` corresponding to this token tree.
    pub fn get_span(&self) -> Span {
        match *self {
            TokenTree::Token(span, _) => span,
            TokenTree::Delimited(span, _) => span,
            TokenTree::Sequence(span, _) => span,
        }
    }

    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt,
                 mtch: &[TokenTree],
                 tts: &[TokenTree])
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let directory = Directory {
            path: cx.current_expansion.module.directory.clone(),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts.iter().cloned().collect(), mtch, Some(directory))
    }

    /// Check if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => tk == tk2,
            (&TokenTree::Delimited(_, ref dl), &TokenTree::Delimited(_, ref dl2)) => {
                (*dl).delim == (*dl2).delim && dl.tts.len() == dl2.tts.len() &&
                {
                    for (tt1, tt2) in dl.tts.iter().zip(dl2.tts.iter()) {
                        if !tt1.eq_unspanned(tt2) {
                            return false;
                        }
                    }
                    true
                }
            }
            (_, _) => false,
        }
    }

    /// Retrieve the TokenTree's span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(sp, _) |
            TokenTree::Delimited(sp, _) |
            TokenTree::Sequence(sp, _) => sp,
        }
    }

    /// Indicates if the stream is a token that is equal to the provided token.
    pub fn eq_token(&self, t: Token) -> bool {
        match *self {
            TokenTree::Token(_, ref tk) => *tk == t,
            _ => false,
        }
    }

    /// Indicates if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        self.maybe_ident().is_some()
    }

    /// Returns an identifier.
    pub fn maybe_ident(&self) -> Option<ast::Ident> {
        match *self {
            TokenTree::Token(_, Token::Ident(t)) => Some(t.clone()),
            TokenTree::Delimited(_, ref dl) => {
                let tts = dl.subtrees();
                if tts.len() != 1 {
                    return None;
                }
                tts[0].maybe_ident()
            }
            _ => None,
        }
    }

    /// Returns a Token literal.
    pub fn maybe_lit(&self) -> Option<token::Lit> {
        match *self {
            TokenTree::Token(_, Token::Literal(l, _)) => Some(l.clone()),
            TokenTree::Delimited(_, ref dl) => {
                let tts = dl.subtrees();
                if tts.len() != 1 {
                    return None;
                }
                tts[0].maybe_lit()
            }
            _ => None,
        }
    }

    /// Returns an AST string literal.
    pub fn maybe_str(&self) -> Option<ast::Lit> {
        match *self {
            TokenTree::Token(sp, Token::Literal(Lit::Str_(s), _)) => {
                let l = LitKind::Str(Symbol::intern(&parse::str_lit(&s.as_str())),
                                     ast::StrStyle::Cooked);
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            TokenTree::Token(sp, Token::Literal(Lit::StrRaw(s, n), _)) => {
                let l = LitKind::Str(Symbol::intern(&parse::raw_str_lit(&s.as_str())),
                                     ast::StrStyle::Raw(n));
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            _ => None,
        }
    }
}

/// #Token Streams
///
/// TokenStreams are a syntactic abstraction over TokenTrees. The goal is for procedural
/// macros to work over TokenStreams instead of arbitrary syntax. For now, however, we
/// are going to cut a few corners (i.e., use some of the AST structure) when we need to
/// for backwards compatibility.

/// TokenStreams are collections of TokenTrees that represent a syntactic structure. The
/// struct itself shouldn't be directly manipulated; the internal structure is not stable,
/// and may be changed at any time in the future. The operators will not, however (except
/// for signatures, later on).
#[derive(Clone, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct TokenStream {
    ts: InternalTS,
}

// This indicates the maximum size for a leaf in the concatenation algorithm.
// If two leafs will be collectively smaller than this, they will be merged.
// If a leaf is larger than this, it will be concatenated at the top.
const LEAF_SIZE : usize = 32;

// NB If Leaf access proves to be slow, inroducing a secondary Leaf without the bounds
// for unsliced Leafs may lead to some performance improvemenet.
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum InternalTS {
    Empty(Span),
    Leaf {
        tts: Rc<Vec<TokenTree>>,
        offset: usize,
        len: usize,
        sp: Span,
    },
    Node {
        left: Rc<InternalTS>,
        right: Rc<InternalTS>,
        len: usize,
        sp: Span,
    },
}

impl fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.ts.fmt(f)
    }
}

impl fmt::Debug for InternalTS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InternalTS::Empty(..) => Ok(()),
            InternalTS::Leaf { ref tts, offset, len, .. } => {
                for t in tts.iter().skip(offset).take(len) {
                    try!(write!(f, "{:?}", t));
                }
                Ok(())
            }
            InternalTS::Node { ref left, ref right, .. } => {
                try!(left.fmt(f));
                right.fmt(f)
            }
        }
    }
}

/// Checks if two TokenStreams are equivalent (including spans). For unspanned
/// equality, see `eq_unspanned`.
impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.iter().eq(other.iter())
    }
}

// NB this will disregard gaps. if we have [a|{2,5} , b|{11,13}], the resultant span
// will be at {2,13}. Without finer-grained span structures, however, this seems to be
// our only recourse.
// FIXME Do something smarter to compute the expansion id.
fn covering_span(trees: &[TokenTree]) -> Span {
    // disregard any dummy spans we have
    let trees = trees.iter().filter(|t| t.span() != DUMMY_SP).collect::<Vec<&TokenTree>>();

    // if we're out of spans, stop
    if trees.len() < 1 {
        return DUMMY_SP;
    }

    // set up the initial values
    let fst_span = trees[0].span();

    let mut lo_span = fst_span.lo;
    let mut hi_span = fst_span.hi;
    let mut expn_id = fst_span.expn_id;

    // compute the spans iteratively
    for t in trees.iter().skip(1) {
        let sp = t.span();
        if sp.lo < lo_span {
            lo_span = sp.lo;
        }
        if hi_span < sp.hi {
            hi_span = sp.hi;
        }
        if expn_id != sp.expn_id {
            expn_id = NO_EXPANSION;
        }
    }

    Span {
        lo: lo_span,
        hi: hi_span,
        expn_id: expn_id,
    }
}

impl InternalTS {
    fn len(&self) -> usize {
        match *self {
            InternalTS::Empty(..) => 0,
            InternalTS::Leaf { len, .. } => len,
            InternalTS::Node { len, .. } => len,
        }
    }

    fn span(&self) -> Span {
        match *self {
            InternalTS::Empty(sp) |
            InternalTS::Leaf { sp, .. } |
            InternalTS::Node { sp, .. } => sp,
        }
    }

    fn slice(&self, range: ops::Range<usize>) -> TokenStream {
        let from = range.start;
        let to = range.end;
        if from == to {
            return TokenStream::mk_empty();
        }
        if from > to {
            panic!("Invalid range: {} to {}", from, to);
        }
        if from == 0 && to == self.len() {
            return TokenStream { ts: self.clone() }; /* should be cheap */
        }
        match *self {
            InternalTS::Empty(..) => panic!("Invalid index"),
            InternalTS::Leaf { ref tts, offset, .. } => {
                let offset = offset + from;
                let len = to - from;
                TokenStream::mk_sub_leaf(tts.clone(),
                                         offset,
                                         len,
                                         covering_span(&tts[offset..offset + len]))
            }
            InternalTS::Node { ref left, ref right, .. } => {
                let left_len = left.len();
                if to <= left_len {
                    left.slice(range)
                } else if from >= left_len {
                    right.slice(from - left_len..to - left_len)
                } else {
                    TokenStream::concat(left.slice(from..left_len), right.slice(0..to - left_len))
                }
            }
        }
    }

    fn to_vec(&self) -> Vec<&TokenTree> {
        let mut res = Vec::with_capacity(self.len());
        fn traverse_and_append<'a>(res: &mut Vec<&'a TokenTree>, ts: &'a InternalTS) {
            match *ts {
                InternalTS::Empty(..) => {},
                InternalTS::Leaf { ref tts, offset, len, .. } => {
                    let mut to_app = tts[offset..offset + len].iter().collect();
                    res.append(&mut to_app);
                }
                InternalTS::Node { ref left, ref right, .. } => {
                    traverse_and_append(res, left);
                    traverse_and_append(res, right);
                }
            }
        }
        traverse_and_append(&mut res, self);
        res
    }

    fn to_tts(&self) -> Vec<TokenTree> {
        self.to_vec().into_iter().cloned().collect::<Vec<TokenTree>>()
    }

    // Returns an internal node's children.
    fn children(&self) -> Option<(Rc<InternalTS>, Rc<InternalTS>)> {
        match *self {
            InternalTS::Node { ref left, ref right, .. } => Some((left.clone(), right.clone())),
            _ => None,
        }
    }
}

/// TokenStream operators include basic destructuring, boolean operations, `maybe_...`
/// operations, and `maybe_..._prefix` operations. Boolean operations are straightforward,
/// indicating information about the structure of the stream. The `maybe_...` operations
/// return `Some<...>` if the tokenstream contains the appropriate item.
///
/// Similarly, the `maybe_..._prefix` operations potentially return a
/// partially-destructured stream as a pair where the first element is the expected item
/// and the second is the remainder of the stream. As anb example,
///
///    `maybe_path_prefix("a::b::c(a,b,c).foo()") -> (a::b::c, "(a,b,c).foo()")`
impl TokenStream {
    // Construct an empty node with a dummy span.
    pub fn mk_empty() -> TokenStream {
        TokenStream { ts: InternalTS::Empty(DUMMY_SP) }
    }

    // Construct an empty node with the provided span.
    fn mk_spanned_empty(sp: Span) -> TokenStream {
        TokenStream { ts: InternalTS::Empty(sp) }
    }

    // Construct a leaf node with a 0 offset and length equivalent to the input.
    fn mk_leaf(tts: Rc<Vec<TokenTree>>, sp: Span) -> TokenStream {
        let len = tts.len();
        TokenStream {
            ts: InternalTS::Leaf {
                tts: tts,
                offset: 0,
                len: len,
                sp: sp,
            },
        }
    }

    // Construct a leaf node with the provided values.
    fn mk_sub_leaf(tts: Rc<Vec<TokenTree>>, offset: usize, len: usize, sp: Span) -> TokenStream {
        TokenStream {
            ts: InternalTS::Leaf {
                tts: tts,
                offset: offset,
                len: len,
                sp: sp,
            },
        }
    }

    // Construct an internal node with the provided values.
    fn mk_int_node(left: Rc<InternalTS>,
                   right: Rc<InternalTS>,
                   len: usize,
                   sp: Span)
                   -> TokenStream {
        TokenStream {
            ts: InternalTS::Node {
                left: left,
                right: right,
                len: len,
                sp: sp,
            },
        }
    }

    /// Convert a vector of `TokenTree`s into a `TokenStream`.
    pub fn from_tts(trees: Vec<TokenTree>) -> TokenStream {
        let span = covering_span(&trees[..]);
        TokenStream::mk_leaf(Rc::new(trees), span)
    }

    /// Convert a vector of Tokens into a TokenStream.
    pub fn from_tokens(tokens: Vec<Token>) -> TokenStream {
        // FIXME do something nicer with the spans
        TokenStream::from_tts(tokens.into_iter().map(|t| TokenTree::Token(DUMMY_SP, t)).collect())
    }

    /// Manually change a TokenStream's span.
    pub fn respan(self, span: Span) -> TokenStream {
        match self.ts {
            InternalTS::Empty(..) => TokenStream::mk_spanned_empty(span),
            InternalTS::Leaf { tts, offset, len, .. } => {
                TokenStream::mk_sub_leaf(tts, offset, len, span)
            }
            InternalTS::Node { left, right, len, .. } => {
                TokenStream::mk_int_node(left, right, len, span)
            }
        }
    }

    /// Concatenates two TokenStreams into a new TokenStream.
    pub fn concat(left: TokenStream, right: TokenStream) -> TokenStream {
        // This internal procedure performs 'aggressive compacting' during concatenation as
        // follows:
        // - If the nodes' combined total total length is less than 32, we copy both of
        //   them into a new vector and build a new leaf node.
        // - If one node is an internal node and the other is a 'small' leaf (length<32),
        //   we recur down the internal node on the appropriate side.
        // - Otherwise, we construct a new internal node that points to them as left and
        // right.
        fn concat_internal(left: Rc<InternalTS>, right: Rc<InternalTS>) -> TokenStream {
            let llen = left.len();
            let rlen = right.len();
            let len = llen + rlen;
            let span = combine_spans(left.span(), right.span());
            if len <= LEAF_SIZE {
                let mut new_vec = left.to_tts();
                let mut rvec = right.to_tts();
                new_vec.append(&mut rvec);
                return TokenStream::mk_leaf(Rc::new(new_vec), span);
            }

            match (left.children(), right.children()) {
                (Some((lleft, lright)), None) => {
                    if rlen <= LEAF_SIZE  {
                        let new_right = concat_internal(lright, right);
                        TokenStream::mk_int_node(lleft, Rc::new(new_right.ts), len, span)
                    } else {
                       TokenStream::mk_int_node(left, right, len, span)
                    }
                }
                (None, Some((rleft, rright))) => {
                    if rlen <= LEAF_SIZE  {
                        let new_left = concat_internal(left, rleft);
                        TokenStream::mk_int_node(Rc::new(new_left.ts), rright, len, span)
                    } else {
                       TokenStream::mk_int_node(left, right, len, span)
                    }
                }
                (_, _) => TokenStream::mk_int_node(left, right, len, span),
            }
        }

        if left.is_empty() {
            right
        } else if right.is_empty() {
            left
        } else {
            concat_internal(Rc::new(left.ts), Rc::new(right.ts))
        }
    }

    /// Indicate if the TokenStream is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a TokenStream's length.
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    /// Convert a TokenStream into a vector of borrowed TokenTrees.
    pub fn to_vec(&self) -> Vec<&TokenTree> {
        self.ts.to_vec()
    }

    /// Convert a TokenStream into a vector of TokenTrees (by cloning the TokenTrees).
    /// (This operation is an O(n) deep copy of the underlying structure.)
    pub fn to_tts(&self) -> Vec<TokenTree> {
        self.ts.to_tts()
    }

    /// Return the TokenStream's span.
    pub fn span(&self) -> Span {
        self.ts.span()
    }

    /// Returns an iterator over a TokenStream (as a sequence of TokenTrees).
    pub fn iter<'a>(&self) -> Iter {
        Iter { vs: self, idx: 0 }
    }

    /// Splits a TokenStream based on the provided `&TokenTree -> bool` predicate.
    pub fn split<P>(&self, pred: P) -> Split<P>
        where P: FnMut(&TokenTree) -> bool
    {
        Split {
            vs: self,
            pred: pred,
            finished: false,
            idx: 0,
        }
    }

    /// Produce a slice of the input TokenStream from the `from` index, inclusive, to the
    /// `to` index, non-inclusive.
    pub fn slice(&self, range: ops::Range<usize>) -> TokenStream {
        self.ts.slice(range)
    }

    /// Slice starting at the provided index, inclusive.
    pub fn slice_from(&self, from: ops::RangeFrom<usize>) -> TokenStream {
        self.slice(from.start..self.len())
    }

    /// Slice up to the provided index, non-inclusive.
    pub fn slice_to(&self, to: ops::RangeTo<usize>) -> TokenStream {
        self.slice(0..to.end)
    }

    /// Indicates where the stream is a single, delimited expression (e.g., `(a,b,c)` or
    /// `{a,b,c}`).
    pub fn is_delimited(&self) -> bool {
        self.maybe_delimited().is_some()
    }

    /// Returns the inside of the delimited term as a new TokenStream.
    pub fn maybe_delimited(&self) -> Option<TokenStream> {
        if !(self.len() == 1) {
            return None;
        }

        // FIXME It would be nice to change Delimited to move the Rc around the TokenTree
        // vector directly in order to avoid the clone here.
        match self[0] {
            TokenTree::Delimited(_, ref rc) => Some(TokenStream::from_tts(rc.tts.clone())),
            _ => None,
        }
    }

    /// Indicates if the stream is exactly one identifier.
    pub fn is_ident(&self) -> bool {
        self.maybe_ident().is_some()
    }

    /// Returns an identifier
    pub fn maybe_ident(&self) -> Option<ast::Ident> {
        if !(self.len() == 1) {
            return None;
        }

        match self[0] {
            TokenTree::Token(_, Token::Ident(t)) => Some(t),
            _ => None,
        }
    }

    /// Compares two TokenStreams, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        for (t1, t2) in self.iter().zip(other.iter()) {
            if !t1.eq_unspanned(t2) {
                return false;
            }
        }
        true
    }

    /// Convert a vector of TokenTrees into a parentheses-delimited TokenStream.
    pub fn as_delimited_stream(tts: Vec<TokenTree>, delim: token::DelimToken) -> TokenStream {
        let new_sp = covering_span(&tts);

        let new_delim = Rc::new(Delimited {
            delim: delim,
            open_span: DUMMY_SP,
            tts: tts,
            close_span: DUMMY_SP,
        });

        TokenStream::from_tts(vec![TokenTree::Delimited(new_sp, new_delim)])
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&pprust::tts_to_string(&self.to_tts()))
    }
}

// FIXME Reimplement this iterator to hold onto a slice iterator for a leaf, getting the
// next leaf's iterator when the current one is exhausted.
pub struct Iter<'a> {
    vs: &'a TokenStream,
    idx: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a TokenTree;

    fn next(&mut self) -> Option<&'a TokenTree> {
        if self.vs.is_empty() || self.idx >= self.vs.len() {
            return None;
        }

        let ret = Some(&self.vs[self.idx]);
        self.idx = self.idx + 1;
        ret
    }
}

pub struct Split<'a, P>
    where P: FnMut(&TokenTree) -> bool
{
    vs: &'a TokenStream,
    pred: P,
    finished: bool,
    idx: usize,
}

impl<'a, P> Iterator for Split<'a, P>
    where P: FnMut(&TokenTree) -> bool
{
    type Item = TokenStream;

    fn next(&mut self) -> Option<TokenStream> {
        if self.finished {
            return None;
        }
        if self.idx >= self.vs.len() {
            self.finished = true;
            return None;
        }

        let mut lookup = self.vs.iter().skip(self.idx);
        match lookup.position(|x| (self.pred)(&x)) {
            None => {
                self.finished = true;
                Some(self.vs.slice_from(self.idx..))
            }
            Some(edx) => {
                let ret = Some(self.vs.slice(self.idx..self.idx + edx));
                self.idx += edx + 1;
                ret
            }
        }
    }
}

impl Index<usize> for TokenStream {
    type Output = TokenTree;

    fn index(&self, index: usize) -> &TokenTree {
        &self.ts[index]
    }
}

impl Index<usize> for InternalTS {
    type Output = TokenTree;

    fn index(&self, index: usize) -> &TokenTree {
        if self.len() <= index {
            panic!("Index {} too large for {:?}", index, self);
        }
        match *self {
            InternalTS::Empty(..) => panic!("Invalid index"),
            InternalTS::Leaf { ref tts, offset, .. } => tts.get(index + offset).unwrap(),
            InternalTS::Node { ref left, ref right, .. } => {
                let left_len = left.len();
                if index < left_len {
                    Index::index(&**left, index)
                } else {
                    Index::index(&**right, index - left_len)
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use syntax::ast::Ident;
    use syntax_pos::{Span, BytePos, NO_EXPANSION, DUMMY_SP};
    use parse::token::{self, Token};
    use util::parser_testing::string_to_tts;
    use std::rc::Rc;

    fn sp(a: u32, b: u32) -> Span {
        Span {
            lo: BytePos(a),
            hi: BytePos(b),
            expn_id: NO_EXPANSION,
        }
    }

    fn as_paren_delimited_stream(tts: Vec<TokenTree>) -> TokenStream {
        TokenStream::as_delimited_stream(tts, token::DelimToken::Paren)
    }

    #[test]
    fn test_concat() {
        let test_res = TokenStream::from_tts(string_to_tts("foo::bar::baz".to_string()));
        let test_fst = TokenStream::from_tts(string_to_tts("foo::bar".to_string()));
        let test_snd = TokenStream::from_tts(string_to_tts("::baz".to_string()));
        let eq_res = TokenStream::concat(test_fst, test_snd);
        assert_eq!(test_res.len(), 5);
        assert_eq!(eq_res.len(), 5);
        assert_eq!(test_res.eq_unspanned(&eq_res), true);
    }

    #[test]
    fn test_from_to_bijection() {
        let test_start = string_to_tts("foo::bar(baz)".to_string());
        let test_end = TokenStream::from_tts(string_to_tts("foo::bar(baz)".to_string())).to_tts();
        assert_eq!(test_start, test_end)
    }

    #[test]
    fn test_to_from_bijection() {
        let test_start = TokenStream::from_tts(string_to_tts("foo::bar(baz)".to_string()));
        let test_end = TokenStream::from_tts(test_start.clone().to_tts());
        assert_eq!(test_start, test_end)
    }

    #[test]
    fn test_eq_0() {
        let test_res = TokenStream::from_tts(string_to_tts("foo".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("foo".to_string()));
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_eq_1() {
        let test_res = TokenStream::from_tts(string_to_tts("::bar::baz".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("::bar::baz".to_string()));
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_eq_2() {
        let test_res = TokenStream::from_tts(string_to_tts("foo::bar".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("foo::bar::baz".to_string()));
        assert_eq!(test_res, test_eqs.slice(0..3))
    }

    #[test]
    fn test_eq_3() {
        let test_res = TokenStream::from_tts(string_to_tts("".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("".to_string()));
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_diseq_0() {
        let test_res = TokenStream::from_tts(string_to_tts("::bar::baz".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("bar::baz".to_string()));
        assert_eq!(test_res == test_eqs, false)
    }

    #[test]
    fn test_diseq_1() {
        let test_res = TokenStream::from_tts(string_to_tts("(bar,baz)".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("bar,baz".to_string()));
        assert_eq!(test_res == test_eqs, false)
    }

    #[test]
    fn test_slice_0() {
        let test_res = TokenStream::from_tts(string_to_tts("foo::bar".to_string()));
        let test_eqs = TokenStream::from_tts(string_to_tts("foo::bar::baz".to_string()));
        assert_eq!(test_res, test_eqs.slice(0..3))
    }

    #[test]
    fn test_slice_1() {
        let test_res = TokenStream::from_tts(string_to_tts("foo::bar::baz".to_string()))
            .slice(2..3);
        let test_eqs = TokenStream::from_tts(vec![TokenTree::Token(sp(5,8),
                                                    token::Ident(Ident::from_str("bar")))]);
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_is_empty() {
        let test0 = TokenStream::from_tts(Vec::new());
        let test1 = TokenStream::from_tts(
            vec![TokenTree::Token(sp(0, 1), Token::Ident(Ident::from_str("a")))]
        );

        let test2 = TokenStream::from_tts(string_to_tts("foo(bar::baz)".to_string()));

        assert_eq!(test0.is_empty(), true);
        assert_eq!(test1.is_empty(), false);
        assert_eq!(test2.is_empty(), false);
    }

    #[test]
    fn test_is_delimited() {
        let test0 = TokenStream::from_tts(string_to_tts("foo(bar::baz)".to_string()));
        let test1 = TokenStream::from_tts(string_to_tts("(bar::baz)".to_string()));
        let test2 = TokenStream::from_tts(string_to_tts("(foo,bar,baz)".to_string()));
        let test3 = TokenStream::from_tts(string_to_tts("(foo,bar,baz)(zab,rab,oof)".to_string()));
        let test4 = TokenStream::from_tts(string_to_tts("(foo,bar,baz)foo".to_string()));
        let test5 = TokenStream::from_tts(string_to_tts("".to_string()));

        assert_eq!(test0.is_delimited(), false);
        assert_eq!(test1.is_delimited(), true);
        assert_eq!(test2.is_delimited(), true);
        assert_eq!(test3.is_delimited(), false);
        assert_eq!(test4.is_delimited(), false);
        assert_eq!(test5.is_delimited(), false);
    }

    #[test]
    fn test_is_ident() {
        let test0 = TokenStream::from_tts(string_to_tts("\"foo\"".to_string()));
        let test1 = TokenStream::from_tts(string_to_tts("5".to_string()));
        let test2 = TokenStream::from_tts(string_to_tts("foo".to_string()));
        let test3 = TokenStream::from_tts(string_to_tts("foo::bar".to_string()));
        let test4 = TokenStream::from_tts(string_to_tts("foo(bar)".to_string()));

        assert_eq!(test0.is_ident(), false);
        assert_eq!(test1.is_ident(), false);
        assert_eq!(test2.is_ident(), true);
        assert_eq!(test3.is_ident(), false);
        assert_eq!(test4.is_ident(), false);
    }

    #[test]
    fn test_maybe_delimited() {
        let test0_input = TokenStream::from_tts(string_to_tts("foo(bar::baz)".to_string()));
        let test1_input = TokenStream::from_tts(string_to_tts("(bar::baz)".to_string()));
        let test2_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)".to_string()));
        let test3_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)(zab,rab)"
            .to_string()));
        let test4_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)foo".to_string()));
        let test5_input = TokenStream::from_tts(string_to_tts("".to_string()));

        let test0 = test0_input.maybe_delimited();
        let test1 = test1_input.maybe_delimited();
        let test2 = test2_input.maybe_delimited();
        let test3 = test3_input.maybe_delimited();
        let test4 = test4_input.maybe_delimited();
        let test5 = test5_input.maybe_delimited();

        assert_eq!(test0, None);

        let test1_expected = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(Ident::from_str("bar"))),
                                       TokenTree::Token(sp(4, 6), token::ModSep),
                                       TokenTree::Token(sp(6, 9),
                                                        token::Ident(Ident::from_str("baz")))]);
        assert_eq!(test1, Some(test1_expected));

        let test2_expected = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(Ident::from_str("foo"))),
                                       TokenTree::Token(sp(4, 5), token::Comma),
                                       TokenTree::Token(sp(5, 8),
                                                        token::Ident(Ident::from_str("bar"))),
                                       TokenTree::Token(sp(8, 9), token::Comma),
                                       TokenTree::Token(sp(9, 12),
                                                        token::Ident(Ident::from_str("baz")))]);
        assert_eq!(test2, Some(test2_expected));

        assert_eq!(test3, None);

        assert_eq!(test4, None);

        assert_eq!(test5, None);
    }

    // pub fn maybe_ident(&self) -> Option<ast::Ident>
    #[test]
    fn test_maybe_ident() {
        let test0 = TokenStream::from_tts(string_to_tts("\"foo\"".to_string())).maybe_ident();
        let test1 = TokenStream::from_tts(string_to_tts("5".to_string())).maybe_ident();
        let test2 = TokenStream::from_tts(string_to_tts("foo".to_string())).maybe_ident();
        let test3 = TokenStream::from_tts(string_to_tts("foo::bar".to_string())).maybe_ident();
        let test4 = TokenStream::from_tts(string_to_tts("foo(bar)".to_string())).maybe_ident();

        assert_eq!(test0, None);
        assert_eq!(test1, None);
        assert_eq!(test2, Some(Ident::from_str("foo")));
        assert_eq!(test3, None);
        assert_eq!(test4, None);
    }

    #[test]
    fn test_as_delimited_stream() {
        let test0 = as_paren_delimited_stream(string_to_tts("foo,bar,".to_string()));
        let test1 = as_paren_delimited_stream(string_to_tts("baz(foo,bar)".to_string()));

        let test0_tts = vec![TokenTree::Token(sp(0, 3), token::Ident(Ident::from_str("foo"))),
                             TokenTree::Token(sp(3, 4), token::Comma),
                             TokenTree::Token(sp(4, 7), token::Ident(Ident::from_str("bar"))),
                             TokenTree::Token(sp(7, 8), token::Comma)];
        let test0_stream = TokenStream::from_tts(vec![TokenTree::Delimited(sp(0, 8),
                                                               Rc::new(Delimited {
                                                                   delim: token::DelimToken::Paren,
                                                                   open_span: DUMMY_SP,
                                                                   tts: test0_tts,
                                                                   close_span: DUMMY_SP,
                                                               }))]);

        assert_eq!(test0, test0_stream);


        let test1_tts = vec![TokenTree::Token(sp(4, 7), token::Ident(Ident::from_str("foo"))),
                             TokenTree::Token(sp(7, 8), token::Comma),
                             TokenTree::Token(sp(8, 11), token::Ident(Ident::from_str("bar")))];

        let test1_parse = vec![TokenTree::Token(sp(0, 3), token::Ident(Ident::from_str("baz"))),
                               TokenTree::Delimited(sp(3, 12),
                                                    Rc::new(Delimited {
                                                        delim: token::DelimToken::Paren,
                                                        open_span: sp(3, 4),
                                                        tts: test1_tts,
                                                        close_span: sp(11, 12),
                                                    }))];

        let test1_stream = TokenStream::from_tts(vec![TokenTree::Delimited(sp(0, 12),
                                                               Rc::new(Delimited {
                                                                   delim: token::DelimToken::Paren,
                                                                   open_span: DUMMY_SP,
                                                                   tts: test1_parse,
                                                                   close_span: DUMMY_SP,
                                                               }))]);

        assert_eq!(test1, test1_stream);
    }
}
