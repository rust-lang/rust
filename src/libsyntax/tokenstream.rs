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
//! A TokenStream also has a slice view, `TokenSlice`, that is analogous to `str` for
//! `String`: it allows the programmer to divvy up, explore, and otherwise partition a
//! TokenStream as borrowed subsequences.

use ast::{self, AttrStyle, LitKind};
use syntax_pos::{Span, DUMMY_SP, NO_EXPANSION};
use codemap::Spanned;
use ext::base;
use ext::tt::macro_parser;
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::lexer;
use parse;
use parse::token::{self, Token, Lit, InternedString, Nonterminal};
use parse::token::Lit as TokLit;

use std::fmt;
use std::mem;
use std::ops::Index;
use std::ops;
use std::iter::*;

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
            TokenTree::Token(_, token::SpecialVarNt(..)) => 2,
            TokenTree::Token(_, token::MatchNt(..)) => 3,
            TokenTree::Token(_, token::Interpolated(Nonterminal::NtTT(..))) => 1,
            TokenTree::Delimited(_, ref delimed) => delimed.tts.len() + 2,
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
                    tts: vec![TokenTree::Token(sp, token::Ident(token::str_to_ident("doc"))),
                              TokenTree::Token(sp, token::Eq),
                              TokenTree::Token(sp, token::Literal(
                                  token::StrRaw(token::intern(&stripped), num_of_hashes), None))],
                    close_span: sp,
                }))
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
            (&TokenTree::Token(sp, token::SpecialVarNt(var)), _) => {
                let v = [TokenTree::Token(sp, token::Dollar),
                         TokenTree::Token(sp, token::Ident(token::str_to_ident(var.as_str())))];
                v[index].clone()
            }
            (&TokenTree::Token(sp, token::MatchNt(name, kind)), _) => {
                let v = [TokenTree::Token(sp, token::SubstNt(name)),
                         TokenTree::Token(sp, token::Colon),
                         TokenTree::Token(sp, token::Ident(kind))];
                v[index].clone()
            }
            (&TokenTree::Token(_, token::Interpolated(Nonterminal::NtTT(ref tt))), _) => {
                tt.clone().unwrap()
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
        let arg_rdr = lexer::new_tt_reader_with_doc_flag(&cx.parse_sess().span_diagnostic,
                                                         None,
                                                         None,
                                                         tts.iter().cloned().collect(),
                                                         true);
        macro_parser::parse(cx.parse_sess(), cx.cfg(), arg_rdr, mtch)
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
                let l = LitKind::Str(token::intern_and_get_ident(&parse::str_lit(&s.as_str())),
                                     ast::StrStyle::Cooked);
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            TokenTree::Token(sp, Token::Literal(Lit::StrRaw(s, n), _)) => {
                let l = LitKind::Str(token::intern_and_get_ident(&parse::raw_str_lit(&s.as_str())),
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
#[derive(Eq,Clone,Hash,RustcEncodable,RustcDecodable)]
pub struct TokenStream {
    pub span: Span,
    pub tts: Vec<TokenTree>,
}

impl fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.tts.len() == 0 {
            write!(f, "([empty")?;
        } else {
            write!(f, "([")?;
            write!(f, "{:?}", self.tts[0])?;

            for tt in self.tts.iter().skip(1) {
                write!(f, ",{:?}", tt)?;
            }
        }
        write!(f, "|")?;
        self.span.fmt(f)?;
        write!(f, "])")
    }
}

/// Checks if two TokenStreams are equivalent (including spans). For unspanned
/// equality, see `eq_unspanned`.
impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.tts == other.tts
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
    /// Convert a vector of `TokenTree`s into a `TokenStream`.
    pub fn from_tts(trees: Vec<TokenTree>) -> TokenStream {
        let span = covering_span(&trees);
        TokenStream {
            tts: trees,
            span: span,
        }
    }

    /// Copies all of the TokenTrees from the TokenSlice, appending them to the stream.
    pub fn append_stream(mut self, ts2: &TokenSlice) {
        for tt in ts2.iter() {
            self.tts.push(tt.clone());
        }
        self.span = covering_span(&self.tts[..]);
    }

    /// Manually change a TokenStream's span.
    pub fn respan(self, span: Span) -> TokenStream {
        TokenStream {
            tts: self.tts,
            span: span,
        }
    }

    /// Construct a TokenStream from an ast literal.
    pub fn from_ast_lit_str(lit: ast::Lit) -> Option<TokenStream> {
        match lit.node {
            LitKind::Str(val, _) => {
                let val = TokLit::Str_(token::intern(&val));
                Some(TokenStream::from_tts(vec![TokenTree::Token(lit.span,
                                                                 Token::Literal(val, None))]))
            }
            _ => None,
        }

    }

    /// Convert a vector of TokenTrees into a parentheses-delimited TokenStream.
    pub fn as_paren_delimited_stream(tts: Vec<TokenTree>) -> TokenStream {
        let new_sp = covering_span(&tts);

        let new_delim = Rc::new(Delimited {
            delim: token::DelimToken::Paren,
            open_span: DUMMY_SP,
            tts: tts,
            close_span: DUMMY_SP,
        });

        TokenStream::from_tts(vec![TokenTree::Delimited(new_sp, new_delim)])
    }

    /// Convert an interned string into a one-element TokenStream.
    pub fn from_interned_string_as_ident(s: InternedString) -> TokenStream {
        TokenStream::from_tts(vec![TokenTree::Token(DUMMY_SP,
                                                    Token::Ident(token::str_to_ident(&s[..])))])
    }
}

/// TokenSlices are 'views' of `TokenStream's; they fit the same role as `str`s do for
/// `String`s. In general, most TokenStream manipulations will be refocusing their internal
/// contents by taking a TokenSlice and then using indexing and the provided operators.
#[derive(PartialEq, Eq, Debug)]
pub struct TokenSlice([TokenTree]);

impl ops::Deref for TokenStream {
    type Target = TokenSlice;

    fn deref(&self) -> &TokenSlice {
        let tts: &[TokenTree] = &*self.tts;
        unsafe { mem::transmute(tts) }
    }
}

impl TokenSlice {
    /// Convert a borrowed TokenTree slice into a borrowed TokenSlice.
    fn from_tts(tts: &[TokenTree]) -> &TokenSlice {
        unsafe { mem::transmute(tts) }
    }

    /// Indicates whether the `TokenStream` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the `TokenSlice`'s length.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check equality versus another TokenStream, ignoring span information.
    pub fn eq_unspanned(&self, other: &TokenSlice) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (tt1, tt2) in self.iter().zip(other.iter()) {
            if !tt1.eq_unspanned(tt2) {
                return false;
            }
        }
        true
    }

    /// Compute a span that covers the entire TokenSlice (eg, one wide enough to include
    /// the entire slice). If the inputs share expansion identification, it is preserved.
    /// If they do not, it is discarded.
    pub fn covering_span(&self) -> Span {
        covering_span(&self.0)
    }

    /// Indicates where the stream is of the form `= <ts>`, where `<ts>` is a continued
    /// `TokenStream`.
    pub fn is_assignment(&self) -> bool {
        self.maybe_assignment().is_some()
    }

    /// Returns the RHS of an assigment.
    pub fn maybe_assignment(&self) -> Option<&TokenSlice> {
        if !(self.len() > 1) {
            return None;
        }

        Some(&self[1..])
    }

    /// Indicates where the stream is a single, delimited expression (e.g., `(a,b,c)` or
    /// `{a,b,c}`).
    pub fn is_delimited(&self) -> bool {
        self.maybe_delimited().is_some()
    }

    /// Returns the inside of the delimited term as a new TokenStream.
    pub fn maybe_delimited(&self) -> Option<&TokenSlice> {
        if !(self.len() == 1) {
            return None;
        }

        match self[0] {
            TokenTree::Delimited(_, ref rc) => Some(TokenSlice::from_tts(&*rc.tts)),
            _ => None,
        }
    }

    /// Returns a list of `TokenSlice`s if the stream is a delimited list, breaking the
    /// stream on commas.
    pub fn maybe_comma_list(&self) -> Option<Vec<&TokenSlice>> {
        let maybe_tts = self.maybe_delimited();

        let ts: &TokenSlice;
        match maybe_tts {
            Some(t) => {
                ts = t;
            }
            None => {
                return None;
            }
        }

        let splits: Vec<&TokenSlice> = ts.split(|x| match *x {
                TokenTree::Token(_, Token::Comma) => true,
                _ => false,
            })
            .filter(|x| x.len() > 0)
            .collect();

        Some(splits)
    }

    /// Returns a Nonterminal if it is Interpolated.
    pub fn maybe_interpolated_nonterminal(&self) -> Option<Nonterminal> {
        if !(self.len() == 1) {
            return None;
        }

        match self[0] {
            TokenTree::Token(_, Token::Interpolated(ref nt)) => Some(nt.clone()),
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

        let tok = if let Some(tts) = self.maybe_delimited() {
            if tts.len() != 1 {
                return None;
            }
            &tts[0]
        } else {
            &self[0]
        };

        match *tok {
            TokenTree::Token(_, Token::Ident(t)) => Some(t),
            _ => None,
        }
    }

    /// Indicates if the stream is exactly one literal
    pub fn is_lit(&self) -> bool {
        self.maybe_lit().is_some()
    }

    /// Returns a literal
    pub fn maybe_lit(&self) -> Option<token::Lit> {
        if !(self.len() == 1) {
            return None;
        }

        let tok = if let Some(tts) = self.maybe_delimited() {
            if tts.len() != 1 {
                return None;
            }
            &tts[0]
        } else {
            &self[0]
        };

        match *tok {
            TokenTree::Token(_, Token::Literal(l, _)) => Some(l),
            _ => None,
        }
    }

    /// Returns an AST string literal if the TokenStream is either a normal ('cooked') or
    /// raw string literal.
    pub fn maybe_str(&self) -> Option<ast::Lit> {
        if !(self.len() == 1) {
            return None;
        }

        match self[0] {
            TokenTree::Token(sp, Token::Literal(Lit::Str_(s), _)) => {
                let l = LitKind::Str(token::intern_and_get_ident(&parse::str_lit(&s.as_str())),
                                     ast::StrStyle::Cooked);
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            TokenTree::Token(sp, Token::Literal(Lit::StrRaw(s, n), _)) => {
                let l = LitKind::Str(token::intern_and_get_ident(&parse::raw_str_lit(&s.as_str())),
                                     ast::StrStyle::Raw(n));
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            _ => None,
        }
    }

    /// This operation extracts the path prefix , returning an AST path struct and the remainder
    /// of the stream (if it finds one). To be more specific, a tokenstream that has a valid,
    /// non-global path as a prefix (eg `foo(bar, baz)`, `foo::bar(bar)`, but *not*
    /// `::foo::bar(baz)`) will yield the path and the remaining tokens (as a slice). The previous
    /// examples will yield
    /// `Some((Path { segments = vec![foo], ... }, [(bar, baz)]))`,
    /// `Some((Path { segments = vec![foo, bar] }, [(baz)]))`,
    /// and `None`, respectively.
    pub fn maybe_path_prefix(&self) -> Option<(ast::Path, &TokenSlice)> {
        let mut segments: Vec<ast::PathSegment> = Vec::new();

        let path: Vec<&TokenTree> = self.iter()
            .take_while(|x| x.is_ident() || x.eq_token(Token::ModSep))
            .collect::<Vec<&TokenTree>>();

        let path_size = path.len();
        if path_size == 0 {
            return None;
        }

        let cov_span = self[..path_size].covering_span();
        let rst = &self[path_size..];

        let fst_id = path[0];

        if let Some(id) = fst_id.maybe_ident() {
            segments.push(ast::PathSegment {
                identifier: id,
                parameters: ast::PathParameters::none(),
            });
        } else {
            return None;
        }

        // Let's use a state machine to parse out the rest.
        enum State {
            Mod, // Expect a `::`, or return None otherwise.
            Ident, // Expect an ident, or return None otherwise.
        }
        let mut state = State::Mod;

        for p in &path[1..] {
            match state {
                State::Mod => {
                    // State 0: ['::' -> state 1, else return None]
                    if p.eq_token(Token::ModSep) {
                        state = State::Ident;
                    } else {
                        return None;
                    }
                }
                State::Ident => {
                    // State 1: [ident -> state 0, else return None]
                    if let Some(id) = p.maybe_ident() {
                        segments.push(ast::PathSegment {
                            identifier: id,
                            parameters: ast::PathParameters::none(),
                        });
                        state = State::Mod;
                    } else {
                        return None;
                    }
                }
            }
        }

        let path = ast::Path {
            span: cov_span,
            global: false,
            segments: segments,
        };
        Some((path, rst))
    }

    /// Returns an iterator over a TokenSlice (as a sequence of TokenStreams).
    fn iter(&self) -> Iter {
        Iter { vs: self }
    }

    /// Splits a TokenSlice based on the provided `&TokenTree -> bool` predicate.
    fn split<P>(&self, pred: P) -> Split<P>
        where P: FnMut(&TokenTree) -> bool
    {
        Split {
            vs: self,
            pred: pred,
            finished: false,
        }
    }
}

pub struct Iter<'a> {
    vs: &'a TokenSlice,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a TokenTree;

    fn next(&mut self) -> Option<&'a TokenTree> {
        if self.vs.is_empty() {
            return None;
        }

        let ret = Some(&self.vs[0]);
        self.vs = &self.vs[1..];
        ret
    }
}

pub struct Split<'a, P>
    where P: FnMut(&TokenTree) -> bool
{
    vs: &'a TokenSlice,
    pred: P,
    finished: bool,
}

impl<'a, P> Iterator for Split<'a, P>
    where P: FnMut(&TokenTree) -> bool
{
    type Item = &'a TokenSlice;

    fn next(&mut self) -> Option<&'a TokenSlice> {
        if self.finished {
            return None;
        }

        match self.vs.iter().position(|x| (self.pred)(x)) {
            None => {
                self.finished = true;
                Some(&self.vs[..])
            }
            Some(idx) => {
                let ret = Some(&self.vs[..idx]);
                self.vs = &self.vs[idx + 1..];
                ret
            }
        }
    }
}

impl Index<usize> for TokenStream {
    type Output = TokenTree;

    fn index(&self, index: usize) -> &TokenTree {
        Index::index(&**self, index)
    }
}

impl ops::Index<ops::Range<usize>> for TokenStream {
    type Output = TokenSlice;

    fn index(&self, index: ops::Range<usize>) -> &TokenSlice {
        Index::index(&**self, index)
    }
}

impl ops::Index<ops::RangeTo<usize>> for TokenStream {
    type Output = TokenSlice;

    fn index(&self, index: ops::RangeTo<usize>) -> &TokenSlice {
        Index::index(&**self, index)
    }
}

impl ops::Index<ops::RangeFrom<usize>> for TokenStream {
    type Output = TokenSlice;

    fn index(&self, index: ops::RangeFrom<usize>) -> &TokenSlice {
        Index::index(&**self, index)
    }
}

impl ops::Index<ops::RangeFull> for TokenStream {
    type Output = TokenSlice;

    fn index(&self, _index: ops::RangeFull) -> &TokenSlice {
        Index::index(&**self, _index)
    }
}

impl Index<usize> for TokenSlice {
    type Output = TokenTree;

    fn index(&self, index: usize) -> &TokenTree {
        &self.0[index]
    }
}

impl ops::Index<ops::Range<usize>> for TokenSlice {
    type Output = TokenSlice;

    fn index(&self, index: ops::Range<usize>) -> &TokenSlice {
        TokenSlice::from_tts(&self.0[index])
    }
}

impl ops::Index<ops::RangeTo<usize>> for TokenSlice {
    type Output = TokenSlice;

    fn index(&self, index: ops::RangeTo<usize>) -> &TokenSlice {
        TokenSlice::from_tts(&self.0[index])
    }
}

impl ops::Index<ops::RangeFrom<usize>> for TokenSlice {
    type Output = TokenSlice;

    fn index(&self, index: ops::RangeFrom<usize>) -> &TokenSlice {
        TokenSlice::from_tts(&self.0[index])
    }
}

impl ops::Index<ops::RangeFull> for TokenSlice {
    type Output = TokenSlice;

    fn index(&self, _index: ops::RangeFull) -> &TokenSlice {
        TokenSlice::from_tts(&self.0[_index])
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ast;
    use syntax_pos::{Span, BytePos, NO_EXPANSION, DUMMY_SP};
    use parse::token::{self, str_to_ident, Token, Lit};
    use util::parser_testing::string_to_tts;
    use std::rc::Rc;

    fn sp(a: u32, b: u32) -> Span {
        Span {
            lo: BytePos(a),
            hi: BytePos(b),
            expn_id: NO_EXPANSION,
        }
    }

    #[test]
    fn test_is_empty() {
        let test0 = TokenStream::from_tts(Vec::new());
        let test1 = TokenStream::from_tts(vec![TokenTree::Token(sp(0, 1),
                                                                Token::Ident(str_to_ident("a")))]);
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
    fn test_is_assign() {
        let test0 = TokenStream::from_tts(string_to_tts("= bar::baz".to_string()));
        let test1 = TokenStream::from_tts(string_to_tts("= \"5\"".to_string()));
        let test2 = TokenStream::from_tts(string_to_tts("= 5".to_string()));
        let test3 = TokenStream::from_tts(string_to_tts("(foo = 10)".to_string()));
        let test4 = TokenStream::from_tts(string_to_tts("= (foo,bar,baz)".to_string()));
        let test5 = TokenStream::from_tts(string_to_tts("".to_string()));

        assert_eq!(test0.is_assignment(), true);
        assert_eq!(test1.is_assignment(), true);
        assert_eq!(test2.is_assignment(), true);
        assert_eq!(test3.is_assignment(), false);
        assert_eq!(test4.is_assignment(), true);
        assert_eq!(test5.is_assignment(), false);
    }

    #[test]
    fn test_is_lit() {
        let test0 = TokenStream::from_tts(string_to_tts("\"foo\"".to_string()));
        let test1 = TokenStream::from_tts(string_to_tts("5".to_string()));
        let test2 = TokenStream::from_tts(string_to_tts("foo".to_string()));
        let test3 = TokenStream::from_tts(string_to_tts("foo::bar".to_string()));
        let test4 = TokenStream::from_tts(string_to_tts("foo(bar)".to_string()));

        assert_eq!(test0.is_lit(), true);
        assert_eq!(test1.is_lit(), true);
        assert_eq!(test2.is_lit(), false);
        assert_eq!(test3.is_lit(), false);
        assert_eq!(test4.is_lit(), false);
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
    fn test_maybe_assignment() {
        let test0_input = TokenStream::from_tts(string_to_tts("= bar::baz".to_string()));
        let test1_input = TokenStream::from_tts(string_to_tts("= \"5\"".to_string()));
        let test2_input = TokenStream::from_tts(string_to_tts("= 5".to_string()));
        let test3_input = TokenStream::from_tts(string_to_tts("(foo = 10)".to_string()));
        let test4_input = TokenStream::from_tts(string_to_tts("= (foo,bar,baz)".to_string()));
        let test5_input = TokenStream::from_tts(string_to_tts("".to_string()));

        let test0 = test0_input.maybe_assignment();
        let test1 = test1_input.maybe_assignment();
        let test2 = test2_input.maybe_assignment();
        let test3 = test3_input.maybe_assignment();
        let test4 = test4_input.maybe_assignment();
        let test5 = test5_input.maybe_assignment();

        let test0_expected = TokenStream::from_tts(vec![TokenTree::Token(sp(2, 5),
                                                        token::Ident(str_to_ident("bar"))),
                                       TokenTree::Token(sp(5, 7), token::ModSep),
                                       TokenTree::Token(sp(7, 10),
                                                        token::Ident(str_to_ident("baz")))]);
        assert_eq!(test0, Some(&test0_expected[..]));

        let test1_expected = TokenStream::from_tts(vec![TokenTree::Token(sp(2, 5),
                                            token::Literal(Lit::Str_(token::intern("5")), None))]);
        assert_eq!(test1, Some(&test1_expected[..]));

        let test2_expected = TokenStream::from_tts(vec![TokenTree::Token( sp(2,3)
                                       , token::Literal(
                                           Lit::Integer(
                                             token::intern(&(5.to_string()))),
                                             None))]);
        assert_eq!(test2, Some(&test2_expected[..]));

        assert_eq!(test3, None);


        let test4_tts = vec![TokenTree::Token(sp(3, 6), token::Ident(str_to_ident("foo"))),
                             TokenTree::Token(sp(6, 7), token::Comma),
                             TokenTree::Token(sp(7, 10), token::Ident(str_to_ident("bar"))),
                             TokenTree::Token(sp(10, 11), token::Comma),
                             TokenTree::Token(sp(11, 14), token::Ident(str_to_ident("baz")))];

        let test4_expected = TokenStream::from_tts(vec![TokenTree::Delimited(sp(2, 15),
                                                Rc::new(Delimited {
                                                    delim: token::DelimToken::Paren,
                                                    open_span: sp(2, 3),
                                                    tts: test4_tts,
                                                    close_span: sp(14, 15),
                                                }))]);
        assert_eq!(test4, Some(&test4_expected[..]));

        assert_eq!(test5, None);

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
                                                        token::Ident(str_to_ident("bar"))),
                                       TokenTree::Token(sp(4, 6), token::ModSep),
                                       TokenTree::Token(sp(6, 9),
                                                        token::Ident(str_to_ident("baz")))]);
        assert_eq!(test1, Some(&test1_expected[..]));

        let test2_expected = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(str_to_ident("foo"))),
                                       TokenTree::Token(sp(4, 5), token::Comma),
                                       TokenTree::Token(sp(5, 8),
                                                        token::Ident(str_to_ident("bar"))),
                                       TokenTree::Token(sp(8, 9), token::Comma),
                                       TokenTree::Token(sp(9, 12),
                                                        token::Ident(str_to_ident("baz")))]);
        assert_eq!(test2, Some(&test2_expected[..]));

        assert_eq!(test3, None);

        assert_eq!(test4, None);

        assert_eq!(test5, None);
    }

    #[test]
    fn test_maybe_comma_list() {
        let test0_input = TokenStream::from_tts(string_to_tts("foo(bar::baz)".to_string()));
        let test1_input = TokenStream::from_tts(string_to_tts("(bar::baz)".to_string()));
        let test2_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)".to_string()));
        let test3_input = TokenStream::from_tts(string_to_tts("(foo::bar,bar,baz)".to_string()));
        let test4_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)(zab,rab)"
            .to_string()));
        let test5_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)foo".to_string()));
        let test6_input = TokenStream::from_tts(string_to_tts("".to_string()));
        // The following is supported behavior!
        let test7_input = TokenStream::from_tts(string_to_tts("(foo,bar,)".to_string()));

        let test0 = test0_input.maybe_comma_list();
        let test1 = test1_input.maybe_comma_list();
        let test2 = test2_input.maybe_comma_list();
        let test3 = test3_input.maybe_comma_list();
        let test4 = test4_input.maybe_comma_list();
        let test5 = test5_input.maybe_comma_list();
        let test6 = test6_input.maybe_comma_list();
        let test7 = test7_input.maybe_comma_list();

        assert_eq!(test0, None);

        let test1_stream = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(str_to_ident("bar"))),
                                       TokenTree::Token(sp(4, 6), token::ModSep),
                                       TokenTree::Token(sp(6, 9),
                                                        token::Ident(str_to_ident("baz")))]);

        let test1_expected: Vec<&TokenSlice> = vec![&test1_stream[..]];
        assert_eq!(test1, Some(test1_expected));

        let test2_foo = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(str_to_ident("foo")))]);
        let test2_bar = TokenStream::from_tts(vec![TokenTree::Token(sp(5, 8),
                                                        token::Ident(str_to_ident("bar")))]);
        let test2_baz = TokenStream::from_tts(vec![TokenTree::Token(sp(9, 12),
                                                        token::Ident(str_to_ident("baz")))]);
        let test2_expected: Vec<&TokenSlice> = vec![&test2_foo[..], &test2_bar[..], &test2_baz[..]];
        assert_eq!(test2, Some(test2_expected));

        let test3_path = TokenStream::from_tts(vec![TokenTree::Token(sp(1, 4),
                                                        token::Ident(str_to_ident("foo"))),
                                       TokenTree::Token(sp(4, 6), token::ModSep),
                                       TokenTree::Token(sp(6, 9),
                                                        token::Ident(str_to_ident("bar")))]);
        let test3_bar = TokenStream::from_tts(vec![TokenTree::Token(sp(10, 13),
                                                        token::Ident(str_to_ident("bar")))]);
        let test3_baz = TokenStream::from_tts(vec![TokenTree::Token(sp(14, 17),
                                                        token::Ident(str_to_ident("baz")))]);
        let test3_expected: Vec<&TokenSlice> =
            vec![&test3_path[..], &test3_bar[..], &test3_baz[..]];
        assert_eq!(test3, Some(test3_expected));

        assert_eq!(test4, None);

        assert_eq!(test5, None);

        assert_eq!(test6, None);


        let test7_expected: Vec<&TokenSlice> = vec![&test2_foo[..], &test2_bar[..]];
        assert_eq!(test7, Some(test7_expected));
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
        assert_eq!(test2, Some(str_to_ident("foo")));
        assert_eq!(test3, None);
        assert_eq!(test4, None);
    }

    // pub fn maybe_lit(&self) -> Option<token::Lit>
    #[test]
    fn test_maybe_lit() {
        let test0 = TokenStream::from_tts(string_to_tts("\"foo\"".to_string())).maybe_lit();
        let test1 = TokenStream::from_tts(string_to_tts("5".to_string())).maybe_lit();
        let test2 = TokenStream::from_tts(string_to_tts("foo".to_string())).maybe_lit();
        let test3 = TokenStream::from_tts(string_to_tts("foo::bar".to_string())).maybe_lit();
        let test4 = TokenStream::from_tts(string_to_tts("foo(bar)".to_string())).maybe_lit();

        assert_eq!(test0, Some(Lit::Str_(token::intern("foo"))));
        assert_eq!(test1, Some(Lit::Integer(token::intern(&(5.to_string())))));
        assert_eq!(test2, None);
        assert_eq!(test3, None);
        assert_eq!(test4, None);
    }

    #[test]
    fn test_maybe_path_prefix() {
        let test0_input = TokenStream::from_tts(string_to_tts("foo(bar::baz)".to_string()));
        let test1_input = TokenStream::from_tts(string_to_tts("(bar::baz)".to_string()));
        let test2_input = TokenStream::from_tts(string_to_tts("(foo,bar,baz)".to_string()));
        let test3_input = TokenStream::from_tts(string_to_tts("foo::bar(bar,baz)".to_string()));

        let test0 = test0_input.maybe_path_prefix();
        let test1 = test1_input.maybe_path_prefix();
        let test2 = test2_input.maybe_path_prefix();
        let test3 = test3_input.maybe_path_prefix();

        let test0_tts = vec![TokenTree::Token(sp(4, 7), token::Ident(str_to_ident("bar"))),
                             TokenTree::Token(sp(7, 9), token::ModSep),
                             TokenTree::Token(sp(9, 12), token::Ident(str_to_ident("baz")))];

        let test0_stream = TokenStream::from_tts(vec![TokenTree::Delimited(sp(3, 13),
                                                               Rc::new(Delimited {
                                                                   delim: token::DelimToken::Paren,
                                                                   open_span: sp(3, 4),
                                                                   tts: test0_tts,
                                                                   close_span: sp(12, 13),
                                                               }))]);

        let test0_expected = Some((ast::Path::from_ident(sp(0, 3), str_to_ident("foo")),
                                   &test0_stream[..]));
        assert_eq!(test0, test0_expected);

        assert_eq!(test1, None);
        assert_eq!(test2, None);

        let test3_path = ast::Path {
            span: sp(0, 8),
            global: false,
            segments: vec![ast::PathSegment {
                               identifier: str_to_ident("foo"),
                               parameters: ast::PathParameters::none(),
                           },
                           ast::PathSegment {
                               identifier: str_to_ident("bar"),
                               parameters: ast::PathParameters::none(),
                           }],
        };

        let test3_tts = vec![TokenTree::Token(sp(9, 12), token::Ident(str_to_ident("bar"))),
                             TokenTree::Token(sp(12, 13), token::Comma),
                             TokenTree::Token(sp(13, 16), token::Ident(str_to_ident("baz")))];

        let test3_stream = TokenStream::from_tts(vec![TokenTree::Delimited(sp(8, 17),
                                                               Rc::new(Delimited {
                                                                   delim: token::DelimToken::Paren,
                                                                   open_span: sp(8, 9),
                                                                   tts: test3_tts,
                                                                   close_span: sp(16, 17),
                                                               }))]);
        let test3_expected = Some((test3_path, &test3_stream[..]));
        assert_eq!(test3, test3_expected);
    }

    #[test]
    fn test_as_paren_delimited_stream() {
        let test0 = TokenStream::as_paren_delimited_stream(string_to_tts("foo,bar,".to_string()));
        let test1 = TokenStream::as_paren_delimited_stream(string_to_tts("baz(foo,bar)"
            .to_string()));

        let test0_tts = vec![TokenTree::Token(sp(0, 3), token::Ident(str_to_ident("foo"))),
                             TokenTree::Token(sp(3, 4), token::Comma),
                             TokenTree::Token(sp(4, 7), token::Ident(str_to_ident("bar"))),
                             TokenTree::Token(sp(7, 8), token::Comma)];
        let test0_stream = TokenStream::from_tts(vec![TokenTree::Delimited(sp(0, 8),
                                                               Rc::new(Delimited {
                                                                   delim: token::DelimToken::Paren,
                                                                   open_span: DUMMY_SP,
                                                                   tts: test0_tts,
                                                                   close_span: DUMMY_SP,
                                                               }))]);

        assert_eq!(test0, test0_stream);


        let test1_tts = vec![TokenTree::Token(sp(4, 7), token::Ident(str_to_ident("foo"))),
                             TokenTree::Token(sp(7, 8), token::Comma),
                             TokenTree::Token(sp(8, 11), token::Ident(str_to_ident("bar")))];

        let test1_parse = vec![TokenTree::Token(sp(0, 3), token::Ident(str_to_ident("baz"))),
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
