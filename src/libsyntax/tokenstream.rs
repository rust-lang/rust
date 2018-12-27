//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves a single `Token` or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//! `TokenStreams` are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a `TokenStream`
//! (such as `slice`) produces an entirely new `TokenStream` from the borrowed reference to
//! the original. This essentially coerces `TokenStream`s into 'views' of their subparts,
//! and a borrowed `TokenStream` is sufficient to build an owned `TokenStream` without taking
//! ownership of the original.

use syntax_pos::{BytePos, Mark, Span, DUMMY_SP};
use ext::base;
use ext::tt::{macro_parser, quoted};
use parse::Directory;
use parse::token::{self, DelimToken, Token};
use print::pprust;
use rustc_data_structures::sync::Lrc;
use serialize::{Decoder, Decodable, Encoder, Encodable};

use std::borrow::Cow;
use std::{fmt, iter, mem};

/// When the main rust parser encounters a syntax-extension invocation, it
/// parses the arguments to the invocation as a token-tree. This is a very
/// loose structure, such that all sorts of different AST-fragments can
/// be passed to syntax extensions using a uniform type.
///
/// If the syntax extension is an MBE macro, it will attempt to match its
/// LHS token tree against the provided token tree, and if it finds a
/// match, will transcribe the RHS token tree, splicing in any captured
/// `macro_parser::matched_nonterminals` into the `SubstNt`s it finds.
///
/// The RHS of an MBE macro is the only place `SubstNt`s are substituted.
/// Nothing special happens to misnamed or misplaced `SubstNt`s.
#[derive(Debug, Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum TokenTree {
    /// A single token
    Token(Span, token::Token),
    /// A delimited sequence of token trees
    Delimited(DelimSpan, DelimToken, ThinTokenStream),
}

impl TokenTree {
    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt, mtch: &[quoted::TokenTree], tts: TokenStream)
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let directory = Directory {
            path: Cow::from(cx.current_expansion.module.directory.as_path()),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts, mtch, Some(directory), true)
    }

    /// Check if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => tk == tk2,
            (&TokenTree::Delimited(_, delim, ref tts),
             &TokenTree::Delimited(_, delim2, ref tts2)) => {
                delim == delim2 &&
                tts.stream().eq_unspanned(&tts2.stream())
            }
            (_, _) => false,
        }
    }

    // See comments in `interpolated_to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => {
                tk.probably_equal_for_proc_macro(tk2)
            }
            (&TokenTree::Delimited(_, delim, ref tts),
             &TokenTree::Delimited(_, delim2, ref tts2)) => {
                delim == delim2 &&
                tts.stream().probably_equal_for_proc_macro(&tts2.stream())
            }
            (_, _) => false,
        }
    }

    /// Retrieve the TokenTree's span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(sp, _) => sp,
            TokenTree::Delimited(sp, ..) => sp.entire(),
        }
    }

    /// Modify the `TokenTree`'s span in-place.
    pub fn set_span(&mut self, span: Span) {
        match *self {
            TokenTree::Token(ref mut sp, _) => *sp = span,
            TokenTree::Delimited(ref mut sp, ..) => *sp = DelimSpan::from_single(span),
        }
    }

    /// Indicates if the stream is a token that is equal to the provided token.
    pub fn eq_token(&self, t: Token) -> bool {
        match *self {
            TokenTree::Token(_, ref tk) => *tk == t,
            _ => false,
        }
    }

    pub fn joint(self) -> TokenStream {
        TokenStream::Tree(self, Joint)
    }

    /// Returns the opening delimiter as a token tree.
    pub fn open_tt(span: Span, delim: DelimToken) -> TokenTree {
        let open_span = if span.is_dummy() {
            span
        } else {
            span.with_hi(span.lo() + BytePos(delim.len() as u32))
        };
        TokenTree::Token(open_span, token::OpenDelim(delim))
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(span: Span, delim: DelimToken) -> TokenTree {
        let close_span = if span.is_dummy() {
            span
        } else {
            span.with_lo(span.hi() - BytePos(delim.len() as u32))
        };
        TokenTree::Token(close_span, token::CloseDelim(delim))
    }
}

/// # Token Streams
///
/// A `TokenStream` is an abstract sequence of tokens, organized into `TokenTree`s.
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `Token::Interpolated` for back-compat.
#[derive(Clone, Debug)]
pub enum TokenStream {
    Empty,
    Tree(TokenTree, IsJoint),
    Stream(Lrc<Vec<TokenStream>>),
}

// `TokenStream` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert!(MEM_SIZE_OF_TOKEN_STREAM: mem::size_of::<TokenStream>() == 32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IsJoint {
    Joint,
    NonJoint
}

use self::IsJoint::*;

impl TokenStream {
    /// Given a `TokenStream` with a `Stream` of only two arguments, return a new `TokenStream`
    /// separating the two arguments with a comma for diagnostic suggestions.
    pub(crate) fn add_comma(&self) -> Option<(TokenStream, Span)> {
        // Used to suggest if a user writes `foo!(a b);`
        if let TokenStream::Stream(ref stream) = self {
            let mut suggestion = None;
            let mut iter = stream.iter().enumerate().peekable();
            while let Some((pos, ts)) = iter.next() {
                if let Some((_, next)) = iter.peek() {
                    let sp = match (&ts, &next) {
                        (TokenStream::Tree(TokenTree::Token(_, token::Token::Comma), NonJoint), _) |
                        (_, TokenStream::Tree(TokenTree::Token(_, token::Token::Comma), NonJoint))
                          => continue,
                        (TokenStream::Tree(TokenTree::Token(sp, _), NonJoint), _) => *sp,
                        (TokenStream::Tree(TokenTree::Delimited(sp, ..), NonJoint), _) =>
                            sp.entire(),
                        _ => continue,
                    };
                    let sp = sp.shrink_to_hi();
                    let comma = TokenStream::Tree(TokenTree::Token(sp, token::Comma), NonJoint);
                    suggestion = Some((pos, comma, sp));
                }
            }
            if let Some((pos, comma, sp)) = suggestion {
                let mut new_stream = vec![];
                let parts = stream.split_at(pos + 1);
                new_stream.extend_from_slice(parts.0);
                new_stream.push(comma);
                new_stream.extend_from_slice(parts.1);
                return Some((TokenStream::new(new_stream), sp));
            }
        }
        None
    }
}

impl From<TokenTree> for TokenStream {
    fn from(tt: TokenTree) -> TokenStream {
        TokenStream::Tree(tt, NonJoint)
    }
}

impl From<Token> for TokenStream {
    fn from(token: Token) -> TokenStream {
        TokenTree::Token(DUMMY_SP, token).into()
    }
}

impl<T: Into<TokenStream>> iter::FromIterator<T> for TokenStream {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        TokenStream::new(iter.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let this = mem::replace(self, TokenStream::Empty);

        // Vector of token streams originally in self.
        let tts: Vec<TokenStream> = match this {
            TokenStream::Empty => {
                let mut vec = Vec::new();
                vec.reserve(iter.size_hint().0);
                vec
            }
            TokenStream::Tree(..) => {
                let mut vec = Vec::new();
                vec.reserve(1 + iter.size_hint().0);
                vec.push(this);
                vec
            }
            TokenStream::Stream(rc_vec) => match Lrc::try_unwrap(rc_vec) {
                Ok(mut vec) => {
                    // Extend in place using the existing capacity if possible.
                    // This is the fast path for libraries like `quote` that
                    // build a token stream.
                    vec.reserve(iter.size_hint().0);
                    vec
                }
                Err(rc_vec) => {
                    // Self is shared so we need to copy and extend that.
                    let mut vec = Vec::new();
                    vec.reserve(rc_vec.len() + iter.size_hint().0);
                    vec.extend_from_slice(&rc_vec);
                    vec
                }
            }
        };

        // Perform the extend, joining tokens as needed along the way.
        let mut builder = TokenStreamBuilder(tts);
        for stream in iter {
            builder.push(stream);
        }

        // Build the resulting token stream. If it contains more than one token,
        // preserve capacity in the vector in anticipation of the caller
        // performing additional calls to extend.
        *self = TokenStream::new(builder.0);
    }
}

impl Eq for TokenStream {}

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.trees().eq(other.trees())
    }
}

impl TokenStream {
    pub fn len(&self) -> usize {
        if let TokenStream::Stream(ref slice) = self {
            slice.len()
        } else {
            0
        }
    }

    pub fn empty() -> TokenStream {
        TokenStream::Empty
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TokenStream::Empty => true,
            _ => false,
        }
    }

    pub fn new(mut streams: Vec<TokenStream>) -> TokenStream {
        match streams.len() {
            0 => TokenStream::empty(),
            1 => streams.pop().unwrap(),
            _ => TokenStream::Stream(Lrc::new(streams)),
        }
    }

    pub fn trees(&self) -> Cursor {
        self.clone().into_trees()
    }

    pub fn into_trees(self) -> Cursor {
        Cursor::new(self)
    }

    /// Compares two TokenStreams, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        let mut t1 = self.trees();
        let mut t2 = other.trees();
        for (t1, t2) in t1.by_ref().zip(t2.by_ref()) {
            if !t1.eq_unspanned(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    // See comments in `interpolated_to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenStream) -> bool {
        // When checking for `probably_eq`, we ignore certain tokens that aren't
        // preserved in the AST. Because they are not preserved, the pretty
        // printer arbitrarily adds or removes them when printing as token
        // streams, making a comparison between a token stream generated from an
        // AST and a token stream which was parsed into an AST more reliable.
        fn semantic_tree(tree: &TokenTree) -> bool {
            match tree {
                // The pretty printer tends to add trailing commas to
                // everything, and in particular, after struct fields.
                | TokenTree::Token(_, Token::Comma)
                // The pretty printer emits `NoDelim` as whitespace.
                | TokenTree::Token(_, Token::OpenDelim(DelimToken::NoDelim))
                | TokenTree::Token(_, Token::CloseDelim(DelimToken::NoDelim))
                // The pretty printer collapses many semicolons into one.
                | TokenTree::Token(_, Token::Semi)
                // The pretty printer collapses whitespace arbitrarily and can
                // introduce whitespace from `NoDelim`.
                | TokenTree::Token(_, Token::Whitespace)
                // The pretty printer can turn `$crate` into `::crate_name`
                | TokenTree::Token(_, Token::ModSep) => false,
                _ => true
            }
        }

        let mut t1 = self.trees().filter(semantic_tree);
        let mut t2 = other.trees().filter(semantic_tree);
        for (t1, t2) in t1.by_ref().zip(t2.by_ref()) {
            if !t1.probably_equal_for_proc_macro(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    /// Precondition: `self` consists of a single token tree.
    /// Returns true if the token tree is a joint operation w.r.t. `proc_macro::TokenNode`.
    pub fn as_tree(self) -> (TokenTree, bool /* joint? */) {
        match self {
            TokenStream::Tree(tree, is_joint) => (tree, is_joint == Joint),
            _ => unreachable!(),
        }
    }

    pub fn map_enumerated<F: FnMut(usize, TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        let mut trees = self.into_trees();
        let mut result = Vec::new();
        let mut i = 0;
        while let Some(stream) = trees.next_as_stream() {
            result.push(match stream {
                TokenStream::Tree(tree, is_joint) => TokenStream::Tree(f(i, tree), is_joint),
                _ => unreachable!()
            });
            i += 1;
        }
        TokenStream::new(result)
    }

    pub fn map<F: FnMut(TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        let mut trees = self.into_trees();
        let mut result = Vec::new();
        while let Some(stream) = trees.next_as_stream() {
            result.push(match stream {
                TokenStream::Tree(tree, is_joint) => TokenStream::Tree(f(tree), is_joint),
                _ => unreachable!()
            });
        }
        TokenStream::new(result)
    }

    fn first_tree_and_joint(&self) -> Option<(TokenTree, IsJoint)> {
        match self {
            TokenStream::Empty => None,
            TokenStream::Tree(ref tree, is_joint) => Some((tree.clone(), *is_joint)),
            TokenStream::Stream(ref stream) => stream.first().unwrap().first_tree_and_joint(),
        }
    }

    fn last_tree_if_joint(&self) -> Option<TokenTree> {
        match self {
            TokenStream::Empty | TokenStream::Tree(_, NonJoint) => None,
            TokenStream::Tree(ref tree, Joint) => Some(tree.clone()),
            TokenStream::Stream(ref stream) => stream.last().unwrap().last_tree_if_joint(),
        }
    }
}

#[derive(Clone)]
pub struct TokenStreamBuilder(Vec<TokenStream>);

impl TokenStreamBuilder {
    pub fn new() -> TokenStreamBuilder {
        TokenStreamBuilder(Vec::new())
    }

    pub fn push<T: Into<TokenStream>>(&mut self, stream: T) {
        let stream = stream.into();
        let last_tree_if_joint = self.0.last().and_then(TokenStream::last_tree_if_joint);
        if let Some(TokenTree::Token(last_span, last_tok)) = last_tree_if_joint {
            if let Some((TokenTree::Token(span, tok), is_joint)) = stream.first_tree_and_joint() {
                if let Some(glued_tok) = last_tok.glue(tok) {
                    let last_stream = self.0.pop().unwrap();
                    self.push_all_but_last_tree(&last_stream);
                    let glued_span = last_span.to(span);
                    let glued_tt = TokenTree::Token(glued_span, glued_tok);
                    let glued_tokenstream = TokenStream::Tree(glued_tt, is_joint);
                    self.0.push(glued_tokenstream);
                    self.push_all_but_first_tree(&stream);
                    return
                }
            }
        }
        self.0.push(stream);
    }

    pub fn add<T: Into<TokenStream>>(mut self, stream: T) -> Self {
        self.push(stream);
        self
    }

    pub fn build(self) -> TokenStream {
        TokenStream::new(self.0)
    }

    fn push_all_but_last_tree(&mut self, stream: &TokenStream) {
        if let TokenStream::Stream(ref streams) = stream {
            let len = streams.len();
            match len {
                1 => {}
                2 => self.0.push(streams[0].clone().into()),
                _ => self.0.push(TokenStream::new(streams[0 .. len - 1].to_vec())),
            }
            self.push_all_but_last_tree(&streams[len - 1])
        }
    }

    fn push_all_but_first_tree(&mut self, stream: &TokenStream) {
        if let TokenStream::Stream(ref streams) = stream {
            let len = streams.len();
            match len {
                1 => {}
                2 => self.0.push(streams[1].clone().into()),
                _ => self.0.push(TokenStream::new(streams[1 .. len].to_vec())),
            }
            self.push_all_but_first_tree(&streams[0])
        }
    }
}

#[derive(Clone)]
pub struct Cursor(CursorKind);

#[derive(Clone)]
enum CursorKind {
    Empty,
    Tree(TokenTree, IsJoint, bool /* consumed? */),
    Stream(StreamCursor),
}

#[derive(Clone)]
struct StreamCursor {
    stream: Lrc<Vec<TokenStream>>,
    index: usize,
    stack: Vec<(Lrc<Vec<TokenStream>>, usize)>,
}

impl StreamCursor {
    fn new(stream: Lrc<Vec<TokenStream>>) -> Self {
        StreamCursor { stream: stream, index: 0, stack: Vec::new() }
    }

    fn next_as_stream(&mut self) -> Option<TokenStream> {
        loop {
            if self.index < self.stream.len() {
                self.index += 1;
                let next = self.stream[self.index - 1].clone();
                match next {
                    TokenStream::Empty => {}
                    TokenStream::Tree(..) => return Some(next),
                    TokenStream::Stream(stream) => self.insert(stream),
                }
            } else if let Some((stream, index)) = self.stack.pop() {
                self.stream = stream;
                self.index = index;
            } else {
                return None;
            }
        }
    }

    fn insert(&mut self, stream: Lrc<Vec<TokenStream>>) {
        self.stack.push((mem::replace(&mut self.stream, stream),
                         mem::replace(&mut self.index, 0)));
    }
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.next_as_stream().map(|stream| match stream {
            TokenStream::Tree(tree, _) => tree,
            _ => unreachable!()
        })
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor(match stream {
            TokenStream::Empty => CursorKind::Empty,
            TokenStream::Tree(tree, is_joint) => CursorKind::Tree(tree, is_joint, false),
            TokenStream::Stream(stream) => CursorKind::Stream(StreamCursor::new(stream)),
        })
    }

    pub fn next_as_stream(&mut self) -> Option<TokenStream> {
        let (stream, consumed) = match self.0 {
            CursorKind::Tree(ref tree, ref is_joint, ref mut consumed @ false) =>
                (TokenStream::Tree(tree.clone(), *is_joint), consumed),
            CursorKind::Stream(ref mut cursor) => return cursor.next_as_stream(),
            _ => return None,
        };

        *consumed = true;
        Some(stream)
    }

    pub fn insert(&mut self, stream: TokenStream) {
        match self.0 {
            _ if stream.is_empty() => return,
            CursorKind::Empty => *self = stream.trees(),
            CursorKind::Tree(_, _, consumed) => {
                *self = TokenStream::new(vec![self.original_stream(), stream]).trees();
                if consumed {
                    self.next();
                }
            }
            CursorKind::Stream(ref mut cursor) => {
                cursor.insert(ThinTokenStream::from(stream).0.unwrap());
            }
        }
    }

    pub fn original_stream(&self) -> TokenStream {
        match self.0 {
            CursorKind::Empty => TokenStream::empty(),
            CursorKind::Tree(ref tree, ref is_joint, _) =>
                TokenStream::Tree(tree.clone(), *is_joint),
            CursorKind::Stream(ref cursor) => TokenStream::Stream(
                cursor.stack.get(0).cloned().map(|(stream, _)| stream)
                    .unwrap_or_else(|| cursor.stream.clone())
            ),
        }
    }

    pub fn look_ahead(&self, n: usize) -> Option<TokenTree> {
        fn look_ahead(streams: &[TokenStream], mut n: usize) -> Result<TokenTree, usize> {
            for stream in streams {
                n = match stream {
                    TokenStream::Tree(ref tree, _) if n == 0 => return Ok(tree.clone()),
                    TokenStream::Tree(..) => n - 1,
                    TokenStream::Stream(ref stream) => match look_ahead(stream, n) {
                        Ok(tree) => return Ok(tree),
                        Err(n) => n,
                    },
                    _ => n,
                };
            }
            Err(n)
        }

        match self.0 {
            CursorKind::Empty |
            CursorKind::Tree(_, _, true) => Err(n),
            CursorKind::Tree(ref tree, _, false) => look_ahead(&[tree.clone().into()], n),
            CursorKind::Stream(ref cursor) => {
                look_ahead(&cursor.stream[cursor.index ..], n).or_else(|mut n| {
                    for &(ref stream, index) in cursor.stack.iter().rev() {
                        n = match look_ahead(&stream[index..], n) {
                            Ok(tree) => return Ok(tree),
                            Err(n) => n,
                        }
                    }

                    Err(n)
                })
            }
        }.ok()
    }
}

/// The `TokenStream` type is large enough to represent a single `TokenTree` without allocation.
/// `ThinTokenStream` is smaller, but needs to allocate to represent a single `TokenTree`.
/// We must use `ThinTokenStream` in `TokenTree::Delimited` to avoid infinite size due to recursion.
#[derive(Debug, Clone)]
pub struct ThinTokenStream(Option<Lrc<Vec<TokenStream>>>);

impl ThinTokenStream {
    pub fn stream(&self) -> TokenStream {
        self.clone().into()
    }
}

impl From<TokenStream> for ThinTokenStream {
    fn from(stream: TokenStream) -> ThinTokenStream {
        ThinTokenStream(match stream {
            TokenStream::Empty => None,
            TokenStream::Tree(..) => Some(Lrc::new(vec![stream])),
            TokenStream::Stream(stream) => Some(stream),
        })
    }
}

impl From<ThinTokenStream> for TokenStream {
    fn from(stream: ThinTokenStream) -> TokenStream {
        stream.0.map(TokenStream::Stream).unwrap_or_else(TokenStream::empty)
    }
}

impl Eq for ThinTokenStream {}

impl PartialEq<ThinTokenStream> for ThinTokenStream {
    fn eq(&self, other: &ThinTokenStream) -> bool {
        TokenStream::from(self.clone()) == TokenStream::from(other.clone())
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&pprust::tokens_to_string(self.clone()))
    }
}

impl Encodable for TokenStream {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        self.trees().collect::<Vec<_>>().encode(encoder)
    }
}

impl Decodable for TokenStream {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<TokenStream, D::Error> {
        Vec::<TokenTree>::decode(decoder).map(|vec| vec.into_iter().collect())
    }
}

impl Encodable for ThinTokenStream {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        TokenStream::from(self.clone()).encode(encoder)
    }
}

impl Decodable for ThinTokenStream {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<ThinTokenStream, D::Error> {
        TokenStream::decode(decoder).map(Into::into)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub struct DelimSpan {
    pub open: Span,
    pub close: Span,
}

impl DelimSpan {
    pub fn from_single(sp: Span) -> Self {
        DelimSpan {
            open: sp,
            close: sp,
        }
    }

    pub fn from_pair(open: Span, close: Span) -> Self {
        DelimSpan { open, close }
    }

    pub fn dummy() -> Self {
        Self::from_single(DUMMY_SP)
    }

    pub fn entire(self) -> Span {
        self.open.with_hi(self.close.hi())
    }

    pub fn apply_mark(self, mark: Mark) -> Self {
        DelimSpan {
            open: self.open.apply_mark(mark),
            close: self.close.apply_mark(mark),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntax::ast::Ident;
    use with_globals;
    use syntax_pos::{Span, BytePos, NO_EXPANSION};
    use parse::token::Token;
    use util::parser_testing::string_to_stream;

    fn string_to_ts(string: &str) -> TokenStream {
        string_to_stream(string.to_owned())
    }

    fn sp(a: u32, b: u32) -> Span {
        Span::new(BytePos(a), BytePos(b), NO_EXPANSION)
    }

    #[test]
    fn test_concat() {
        with_globals(|| {
            let test_res = string_to_ts("foo::bar::baz");
            let test_fst = string_to_ts("foo::bar");
            let test_snd = string_to_ts("::baz");
            let eq_res = TokenStream::new(vec![test_fst, test_snd]);
            assert_eq!(test_res.trees().count(), 5);
            assert_eq!(eq_res.trees().count(), 5);
            assert_eq!(test_res.eq_unspanned(&eq_res), true);
        })
    }

    #[test]
    fn test_to_from_bijection() {
        with_globals(|| {
            let test_start = string_to_ts("foo::bar(baz)");
            let test_end = test_start.trees().collect();
            assert_eq!(test_start, test_end)
        })
    }

    #[test]
    fn test_eq_0() {
        with_globals(|| {
            let test_res = string_to_ts("foo");
            let test_eqs = string_to_ts("foo");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_1() {
        with_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("::bar::baz");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_3() {
        with_globals(|| {
            let test_res = string_to_ts("");
            let test_eqs = string_to_ts("");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_diseq_0() {
        with_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("bar::baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_diseq_1() {
        with_globals(|| {
            let test_res = string_to_ts("(bar,baz)");
            let test_eqs = string_to_ts("bar,baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_is_empty() {
        with_globals(|| {
            let test0: TokenStream = Vec::<TokenTree>::new().into_iter().collect();
            let test1: TokenStream =
                TokenTree::Token(sp(0, 1), Token::Ident(Ident::from_str("a"), false)).into();
            let test2 = string_to_ts("foo(bar::baz)");

            assert_eq!(test0.is_empty(), true);
            assert_eq!(test1.is_empty(), false);
            assert_eq!(test2.is_empty(), false);
        })
    }

    #[test]
    fn test_dotdotdot() {
        let mut builder = TokenStreamBuilder::new();
        builder.push(TokenTree::Token(sp(0, 1), Token::Dot).joint());
        builder.push(TokenTree::Token(sp(1, 2), Token::Dot).joint());
        builder.push(TokenTree::Token(sp(2, 3), Token::Dot));
        let stream = builder.build();
        assert!(stream.eq_unspanned(&string_to_ts("...")));
        assert_eq!(stream.trees().count(), 1);
    }

    #[test]
    fn test_extend_empty() {
        with_globals(|| {
            // Append a token onto an empty token stream.
            let mut stream = TokenStream::empty();
            stream.extend(vec![string_to_ts("t")]);

            let expected = string_to_ts("t");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_nothing() {
        with_globals(|| {
            // Append nothing onto a token stream containing one token.
            let mut stream = string_to_ts("t");
            stream.extend(vec![]);

            let expected = string_to_ts("t");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_single() {
        with_globals(|| {
            // Append a token onto token stream containing a single token.
            let mut stream = string_to_ts("t1");
            stream.extend(vec![string_to_ts("t2")]);

            let expected = string_to_ts("t1 t2");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_in_place() {
        with_globals(|| {
            // Append a token onto token stream containing a reference counted
            // vec of tokens. The token stream has a reference count of 1 so
            // this can happen in place.
            let mut stream = string_to_ts("t1 t2");
            stream.extend(vec![string_to_ts("t3")]);

            let expected = string_to_ts("t1 t2 t3");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_copy() {
        with_globals(|| {
            // Append a token onto token stream containing a reference counted
            // vec of tokens. The token stream is shared so the extend takes
            // place on a copy.
            let mut stream = string_to_ts("t1 t2");
            let _incref = stream.clone();
            stream.extend(vec![string_to_ts("t3")]);

            let expected = string_to_ts("t1 t2 t3");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_no_join() {
        with_globals(|| {
            let first = TokenTree::Token(DUMMY_SP, Token::Dot);
            let second = TokenTree::Token(DUMMY_SP, Token::Dot);

            // Append a dot onto a token stream containing a dot, but do not
            // join them.
            let mut stream = TokenStream::from(first);
            stream.extend(vec![TokenStream::from(second)]);

            let expected = string_to_ts(". .");
            assert!(stream.eq_unspanned(&expected));

            let unexpected = string_to_ts("..");
            assert!(!stream.eq_unspanned(&unexpected));
        });
    }

    #[test]
    fn test_extend_join() {
        with_globals(|| {
            let first = TokenTree::Token(DUMMY_SP, Token::Dot).joint();
            let second = TokenTree::Token(DUMMY_SP, Token::Dot);

            // Append a dot onto a token stream containing a dot, forming a
            // dotdot.
            let mut stream = first;
            stream.extend(vec![TokenStream::from(second)]);

            let expected = string_to_ts("..");
            assert!(stream.eq_unspanned(&expected));

            let unexpected = string_to_ts(". .");
            assert!(!stream.eq_unspanned(&unexpected));
        });
    }
}
