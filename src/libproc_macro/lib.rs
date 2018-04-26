// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A support library for macro authors when defining new macros.
//!
//! This library, provided by the standard distribution, provides the types
//! consumed in the interfaces of procedurally defined macro definitions.
//! Currently the primary use of this crate is to provide the ability to define
//! new custom derive modes through `#[proc_macro_derive]`.
//!
//! Note that this crate is intentionally very bare-bones currently. The main
//! type, `TokenStream`, only supports `fmt::Display` and `FromStr`
//! implementations, indicating that it can only go to and come from a string.
//! This functionality is intended to be expanded over time as more surface
//! area for macro authors is stabilized.
//!
//! See [the book](../book/first-edition/procedural-macros.html) for more.

#![stable(feature = "proc_macro_lib", since = "1.15.0")]
#![deny(missing_docs)]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables, unused_mut))))]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(lang_items)]
#![feature(optin_builtin_traits)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors;
extern crate rustc_data_structures;

mod diagnostic;

#[unstable(feature = "proc_macro", issue = "38356")]
pub use diagnostic::{Diagnostic, Level};

use std::{ascii, fmt, iter};
use rustc_data_structures::sync::Lrc;
use std::str::FromStr;

use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::parse::{self, token};
use syntax::symbol::Symbol;
use syntax::tokenstream;
use syntax::parse::lexer::comments;
use syntax_pos::{FileMap, Pos, SyntaxContext, FileName};
use syntax_pos::hygiene::Mark;

/// The main type provided by this crate, representing an abstract stream of
/// tokens.
///
/// This is both the input and output of `#[proc_macro_derive]` definitions.
/// Currently it's required to be a list of valid Rust items, but this
/// restriction may be lifted in the future.
///
/// The API of this type is intentionally bare-bones, but it'll be expanded over
/// time!
#[stable(feature = "proc_macro_lib", since = "1.15.0")]
#[derive(Clone)]
pub struct TokenStream(tokenstream::TokenStream);

/// Error returned from `TokenStream::from_str`.
#[stable(feature = "proc_macro_lib", since = "1.15.0")]
#[derive(Debug)]
pub struct LexError {
    _inner: (),
}

impl TokenStream {
    /// Returns an empty `TokenStream`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn empty() -> TokenStream {
        TokenStream(tokenstream::TokenStream::empty())
    }

    /// Checks if this `TokenStream` is empty.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        __internal::with_sess(|(sess, mark)| {
            let src = src.to_string();
            let name = FileName::ProcMacroSourceCode;
            let expn_info = mark.expn_info().unwrap();
            let call_site = expn_info.call_site;
            // notify the expansion info that it is unhygienic
            let mark = Mark::fresh(mark);
            mark.set_expn_info(expn_info);
            let span = call_site.with_ctxt(SyntaxContext::empty().apply_mark(mark));
            let stream = parse::parse_stream_from_source_str(name, src, sess, Some(span));
            Ok(__internal::token_stream_wrap(stream))
        })
    }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("TokenStream ")?;
        f.debug_list().entries(self.clone()).finish()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream(tree.to_internal())
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl iter::FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(trees: I) -> Self {
        trees.into_iter().map(TokenStream::from).collect()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl iter::FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        let mut builder = tokenstream::TokenStreamBuilder::new();
        for stream in streams {
            builder.push(stream.0);
        }
        TokenStream(builder.build())
    }
}

/// Implementation details for the `TokenTree` type, such as iterators.
#[unstable(feature = "proc_macro", issue = "38356")]
pub mod token_stream {
    use syntax::tokenstream;
    use syntax_pos::DUMMY_SP;

    use {TokenTree, TokenStream, Delimiter};

    /// An iterator over `TokenTree`s.
    #[derive(Clone)]
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub struct IntoIter {
        cursor: tokenstream::Cursor,
        stack: Vec<TokenTree>,
    }

    #[unstable(feature = "proc_macro", issue = "38356")]
    impl Iterator for IntoIter {
        type Item = TokenTree;

        fn next(&mut self) -> Option<TokenTree> {
            loop {
                let tree = self.stack.pop().or_else(|| {
                    let next = self.cursor.next_as_stream()?;
                    Some(TokenTree::from_internal(next, &mut self.stack))
                })?;
                if tree.span().0 == DUMMY_SP {
                    if let TokenTree::Group(ref group) = tree {
                        if group.delimiter() == Delimiter::None {
                            self.cursor.insert(group.stream.clone().0);
                            continue
                        }
                    }
                }
                return Some(tree);
            }
        }
    }

    #[unstable(feature = "proc_macro", issue = "38356")]
    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = IntoIter;

        fn into_iter(self) -> IntoIter {
            IntoIter { cursor: self.0.trees(), stack: Vec::new() }
        }
    }
}

/// `quote!(..)` accepts arbitrary tokens and expands into a `TokenStream` describing the input.
/// For example, `quote!(a + b)` will produce a expression, that, when evaluated, constructs
/// the `TokenStream` `[Word("a"), Op('+', Alone), Word("b")]`.
///
/// Unquoting is done with `$`, and works by taking the single next ident as the unquoted term.
/// To quote `$` itself, use `$$`.
#[unstable(feature = "proc_macro", issue = "38356")]
#[macro_export]
macro_rules! quote { () => {} }

#[unstable(feature = "proc_macro_internals", issue = "27812")]
#[doc(hidden)]
mod quote;

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
#[unstable(feature = "proc_macro", issue = "38356")]
pub fn quote_span(span: Span) -> TokenStream {
    quote::Quote::quote(span)
}

/// A region of source code, along with macro expansion information.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Copy, Clone)]
pub struct Span(syntax_pos::Span);

macro_rules! diagnostic_method {
    ($name:ident, $level:expr) => (
        /// Create a new `Diagnostic` with the given `message` at the span
        /// `self`.
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $name<T: Into<String>>(self, message: T) -> Diagnostic {
            Diagnostic::spanned(self, $level, message)
        }
    )
}

impl Span {
    /// A span that resolves at the macro definition site.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn def_site() -> Span {
        ::__internal::with_sess(|(_, mark)| {
            let call_site = mark.expn_info().unwrap().call_site;
            Span(call_site.with_ctxt(SyntaxContext::empty().apply_mark(mark)))
        })
    }

    /// The span of the invocation of the current procedural macro.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn call_site() -> Span {
        ::__internal::with_sess(|(_, mark)| Span(mark.expn_info().unwrap().call_site))
    }

    /// The original source file into which this span points.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn source_file(&self) -> SourceFile {
        SourceFile {
            filemap: __internal::lookup_char_pos(self.0.lo()).file,
        }
    }

    /// The `Span` for the tokens in the previous macro expansion from which
    /// `self` was generated from, if any.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn parent(&self) -> Option<Span> {
        self.0.parent().map(Span)
    }

    /// The span for the origin source code that `self` was generated from. If
    /// this `Span` wasn't generated from other macro expansions then the return
    /// value is the same as `*self`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn source(&self) -> Span {
        Span(self.0.source_callsite())
    }

    /// Get the starting line/column in the source file for this span.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn start(&self) -> LineColumn {
        let loc = __internal::lookup_char_pos(self.0.lo());
        LineColumn {
            line: loc.line,
            column: loc.col.to_usize()
        }
    }

    /// Get the ending line/column in the source file for this span.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn end(&self) -> LineColumn {
        let loc = __internal::lookup_char_pos(self.0.hi());
        LineColumn {
            line: loc.line,
            column: loc.col.to_usize()
        }
    }

    /// Create a new span encompassing `self` and `other`.
    ///
    /// Returns `None` if `self` and `other` are from different files.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn join(&self, other: Span) -> Option<Span> {
        let self_loc = __internal::lookup_char_pos(self.0.lo());
        let other_loc = __internal::lookup_char_pos(other.0.lo());

        if self_loc.file.name != other_loc.file.name { return None }

        Some(Span(self.0.to(other.0)))
    }

    /// Creates a new span with the same line/column information as `self` but
    /// that resolves symbols as though it were at `other`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn resolved_at(&self, other: Span) -> Span {
        Span(self.0.with_ctxt(other.0.ctxt()))
    }

    /// Creates a new span with the same name resolution behavior as `self` but
    /// with the line/column information of `other`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn located_at(&self, other: Span) -> Span {
        other.resolved_at(*self)
    }

    /// Compares to spans to see if they're equal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn eq(&self, other: &Span) -> bool {
        self.0 == other.0
    }

    diagnostic_method!(error, Level::Error);
    diagnostic_method!(warning, Level::Warning);
    diagnostic_method!(note, Level::Note);
    diagnostic_method!(help, Level::Help);
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} bytes({}..{})",
               self.0.ctxt(),
               self.0.lo().0,
               self.0.hi().0)
    }
}

/// A line-column pair representing the start or end of a `Span`.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineColumn {
    /// The 1-indexed line in the source file on which the span starts or ends (inclusive).
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub line: usize,
    /// The 0-indexed column (in UTF-8 characters) in the source file on which
    /// the span starts or ends (inclusive).
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub column: usize
}

/// The source file of a given `Span`.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Clone)]
pub struct SourceFile {
    filemap: Lrc<FileMap>,
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl !Send for SourceFile {}
#[unstable(feature = "proc_macro", issue = "38356")]
impl !Sync for SourceFile {}

impl SourceFile {
    /// Get the path to this source file.
    ///
    /// ### Note
    /// If the code span associated with this `SourceFile` was generated by an external macro, this
    /// may not be an actual path on the filesystem. Use [`is_real`] to check.
    ///
    /// Also note that even if `is_real` returns `true`, if `--remap-path-prefix` was passed on
    /// the command line, the path as given may not actually be valid.
    ///
    /// [`is_real`]: #method.is_real
    # [unstable(feature = "proc_macro", issue = "38356")]
    pub fn path(&self) -> &FileName {
        &self.filemap.name
    }

    /// Returns `true` if this source file is a real source file, and not generated by an external
    /// macro's expansion.
    # [unstable(feature = "proc_macro", issue = "38356")]
    pub fn is_real(&self) -> bool {
        // This is a hack until intercrate spans are implemented and we can have real source files
        // for spans generated in external macros.
        // https://github.com/rust-lang/rust/pull/43604#issuecomment-333334368
        self.filemap.is_real_file()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl AsRef<FileName> for SourceFile {
    fn as_ref(&self) -> &FileName {
        self.path()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Debug for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SourceFile")
            .field("path", self.path())
            .field("is_real", &self.is_real())
            .finish()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl PartialEq for SourceFile {
    fn eq(&self, other: &Self) -> bool {
        Lrc::ptr_eq(&self.filemap, &other.filemap)
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl Eq for SourceFile {}

#[unstable(feature = "proc_macro", issue = "38356")]
impl PartialEq<FileName> for SourceFile {
    fn eq(&self, other: &FileName) -> bool {
        self.as_ref() == other
    }
}

/// A single token or a delimited sequence of token trees (e.g. `[1, (), ..]`).
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Clone)]
pub enum TokenTree {
    /// A delimited tokenstream
    Group(Group),
    /// A unicode identifier
    Term(Term),
    /// A punctuation character (`+`, `,`, `$`, etc.).
    Op(Op),
    /// A literal character (`'a'`), string (`"hello"`), number (`2.3`), etc.
    Literal(Literal),
}

impl TokenTree {
    /// Returns the span of this token, accessing the `span` method of each of
    /// the internal tokens.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Group(ref t) => t.span(),
            TokenTree::Term(ref t) => t.span(),
            TokenTree::Op(ref t) => t.span(),
            TokenTree::Literal(ref t) => t.span(),
        }
    }

    /// Configures the span for *only this token*.
    ///
    /// Note that if this token is a `Group` then this method will not configure
    /// the span of each of the internal tokens, this will simply delegate to
    /// the `set_span` method of each variant.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        match *self {
            TokenTree::Group(ref mut t) => t.set_span(span),
            TokenTree::Term(ref mut t) => t.set_span(span),
            TokenTree::Op(ref mut t) => t.set_span(span),
            TokenTree::Literal(ref mut t) => t.set_span(span),
        }
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Debug for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Each of these has the name in the struct type in the derived debug,
        // so don't bother with an extra layer of indirection
        match *self {
            TokenTree::Group(ref tt) => tt.fmt(f),
            TokenTree::Term(ref tt) => tt.fmt(f),
            TokenTree::Op(ref tt) => tt.fmt(f),
            TokenTree::Literal(ref tt) => tt.fmt(f),
        }
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<Group> for TokenTree {
    fn from(g: Group) -> TokenTree {
        TokenTree::Group(g)
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<Term> for TokenTree {
    fn from(g: Term) -> TokenTree {
        TokenTree::Term(g)
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<Op> for TokenTree {
    fn from(g: Op) -> TokenTree {
        TokenTree::Op(g)
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<Literal> for TokenTree {
    fn from(g: Literal) -> TokenTree {
        TokenTree::Literal(g)
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TokenTree::Group(ref t) => t.fmt(f),
            TokenTree::Term(ref t) => t.fmt(f),
            TokenTree::Op(ref t) => t.fmt(f),
            TokenTree::Literal(ref t) => t.fmt(f),
        }
    }
}

/// A delimited token stream
///
/// A `Group` internally contains a `TokenStream` which is delimited by a
/// `Delimiter`. Groups represent multiple tokens internally and have a `Span`
/// for the entire stream.
#[derive(Clone, Debug)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct Group {
    delimiter: Delimiter,
    stream: TokenStream,
    span: Span,
}

/// Describes how a sequence of token trees is delimited.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    /// An implicit delimiter, e.g. `$var`, where $var is  `...`.
    None,
}

impl Group {
    /// Creates a new `group` with the given delimiter and token stream.
    ///
    /// This constructor will set the span for this group to
    /// `Span::call_site()`. To change the span you can use the `set_span`
    /// method below.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn new(delimiter: Delimiter, stream: TokenStream) -> Group {
        Group {
            delimiter: delimiter,
            stream: stream,
            span: Span::call_site(),
        }
    }

    /// Returns the delimiter of this `Group`
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn delimiter(&self) -> Delimiter {
        self.delimiter
    }

    /// Returns the `TokenStream` of tokens that are delimited in this `Group`.
    ///
    /// Note that the returned token stream does not include the delimiter
    /// returned above.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn stream(&self) -> TokenStream {
        self.stream.clone()
    }

    /// Returns the span for the delimiters of this token stream, spanning the
    /// entire `Group`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configures the span for this `Group`'s delimiters, but not its internal
    /// tokens.
    ///
    /// This method will **not** set the span of all the internal tokens spanned
    /// by this group, but rather it will only set the span of the delimiter
    /// tokens at the level of the `Group`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        TokenStream::from(TokenTree::from(self.clone())).fmt(f)
    }
}

/// An `Op` is an operator like `+` or `-`, and only represents one character.
///
/// Operators like `+=` are represented as two instance of `Op` with different
/// forms of `Spacing` returned.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Copy, Clone, Debug)]
pub struct Op {
    op: char,
    spacing: Spacing,
    span: Span,
}

/// Whether an `Op` is either followed immediately by another `Op` or followed by whitespace.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub enum Spacing {
    /// e.g. `+` is `Alone` in `+ =`.
    Alone,
    /// e.g. `+` is `Joint` in `+=`.
    Joint,
}

impl Op {
    /// Creates a new `Op` from the given character and spacing.
    ///
    /// The returned `Op` will have the default span of `Span::call_site()`
    /// which can be further configured with the `set_span` method below.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn new(op: char, spacing: Spacing) -> Op {
        Op {
            op: op,
            spacing: spacing,
            span: Span::call_site(),
        }
    }

    /// Returns the character this operation represents, for example `'+'`
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn op(&self) -> char {
        self.op
    }

    /// Returns the spacing of this operator, indicating whether it's a joint
    /// operator with more operators coming next in the token stream or an
    /// `Alone` meaning that the operator has ended.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn spacing(&self) -> Spacing {
        self.spacing
    }

    /// Returns the span for this operator character
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configure the span for this operator's character
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        TokenStream::from(TokenTree::from(self.clone())).fmt(f)
    }
}

/// An interned string.
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct Term {
    sym: Symbol,
    span: Span,
}

impl Term {
    /// Creates a new `Term` with the given `string` as well as the specified
    /// `span`.
    ///
    /// Note that `span`, currently in rustc, configures the hygiene information
    /// for this identifier. As of this time `Span::call_site()` explicitly
    /// opts-in to **non-hygienic** information (aka copy/pasted code) while
    /// spans like `Span::def_site()` will opt-in to hygienic information,
    /// meaning that code at the call site of the macro can't access this
    /// identifier.
    ///
    /// Due to the current importance of hygiene this constructor, unlike other
    /// tokens, requires a `Span` to be specified at construction.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn new(string: &str, span: Span) -> Term {
        Term {
            sym: Symbol::intern(string),
            span,
        }
    }

    /// Get a reference to the interned string.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn as_str(&self) -> &str {
        unsafe { &*(&*self.sym.as_str() as *const str) }
    }

    /// Returns the span of this `Term`, encompassing the entire string returned
    /// by `as_str`.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configures the span of this `Term`, possibly changing hygiene
    /// information.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

/// A literal character (`'a'`), string (`"hello"`), a number (`2.3`), etc.
#[derive(Clone, Debug)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct Literal {
    lit: token::Lit,
    suffix: Option<ast::Name>,
    span: Span,
}

macro_rules! suffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new suffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1u32` where the integer
        /// value specified is the first part of the token and the integral is
        /// also suffixed at the end.
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $name(n: $kind) -> Literal {
            Literal {
                lit: token::Lit::Integer(Symbol::intern(&n.to_string())),
                suffix: Some(Symbol::intern(stringify!($kind))),
                span: Span::call_site(),
            }
        }
    )*)
}

macro_rules! unsuffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new unsuffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1` where the integer
        /// value specified is the first part of the token. No suffix is
        /// specified on this token, meaning that invocations like
        /// `Literal::i8_unsuffixed(1)` are equivalent to
        /// `Literal::u32_unsuffixed(1)`.
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $name(n: $kind) -> Literal {
            Literal {
                lit: token::Lit::Integer(Symbol::intern(&n.to_string())),
                suffix: None,
                span: Span::call_site(),
            }
        }
    )*)
}

impl Literal {
    suffixed_int_literals! {
        u8_suffixed => u8,
        u16_suffixed => u16,
        u32_suffixed => u32,
        u64_suffixed => u64,
        u128_suffixed => u128,
        usize_suffixed => usize,
        i8_suffixed => i8,
        i16_suffixed => i16,
        i32_suffixed => i32,
        i64_suffixed => i64,
        i128_suffixed => i128,
        isize_suffixed => isize,
    }

    unsuffixed_int_literals! {
        u8_unsuffixed => u8,
        u16_unsuffixed => u16,
        u32_unsuffixed => u32,
        u64_unsuffixed => u64,
        u128_unsuffixed => u128,
        usize_unsuffixed => usize,
        i8_unsuffixed => i8,
        i16_unsuffixed => i16,
        i32_unsuffixed => i32,
        i64_unsuffixed => i64,
        i128_unsuffixed => i128,
        isize_unsuffixed => isize,
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f32_unsuffixed(n: f32) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal {
            lit: token::Lit::Float(Symbol::intern(&n.to_string())),
            suffix: None,
            span: Span::call_site(),
        }
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This consturctor will create a literal like `1.0f32` where the value
    /// specified is the preceding part of the token and `f32` is the suffix of
    /// the token. This token will always be inferred to be an `f32` in the
    /// compiler.
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f32_suffixed(n: f32) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal {
            lit: token::Lit::Float(Symbol::intern(&n.to_string())),
            suffix: Some(Symbol::intern("f32")),
            span: Span::call_site(),
        }
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f64_unsuffixed(n: f64) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal {
            lit: token::Lit::Float(Symbol::intern(&n.to_string())),
            suffix: None,
            span: Span::call_site(),
        }
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This consturctor will create a literal like `1.0f64` where the value
    /// specified is the preceding part of the token and `f64` is the suffix of
    /// the token. This token will always be inferred to be an `f64` in the
    /// compiler.
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f64_suffixed(n: f64) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal {
            lit: token::Lit::Float(Symbol::intern(&n.to_string())),
            suffix: Some(Symbol::intern("f64")),
            span: Span::call_site(),
        }
    }

    /// String literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn string(string: &str) -> Literal {
        let mut escaped = String::new();
        for ch in string.chars() {
            escaped.extend(ch.escape_debug());
        }
        Literal {
            lit: token::Lit::Str_(Symbol::intern(&escaped)),
            suffix: None,
            span: Span::call_site(),
        }
    }

    /// Character literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn character(ch: char) -> Literal {
        let mut escaped = String::new();
        escaped.extend(ch.escape_unicode());
        Literal {
            lit: token::Lit::Char(Symbol::intern(&escaped)),
            suffix: None,
            span: Span::call_site(),
        }
    }

    /// Byte string literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn byte_string(bytes: &[u8]) -> Literal {
        let string = bytes.iter().cloned().flat_map(ascii::escape_default)
            .map(Into::<char>::into).collect::<String>();
        Literal {
            lit: token::Lit::ByteStr(Symbol::intern(&string)),
            suffix: None,
            span: Span::call_site(),
        }
    }

    /// Returns the span encompassing this literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configures the span associated for this literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        TokenStream::from(TokenTree::from(self.clone())).fmt(f)
    }
}

impl Delimiter {
    fn from_internal(delim: token::DelimToken) -> Delimiter {
        match delim {
            token::Paren => Delimiter::Parenthesis,
            token::Brace => Delimiter::Brace,
            token::Bracket => Delimiter::Bracket,
            token::NoDelim => Delimiter::None,
        }
    }

    fn to_internal(self) -> token::DelimToken {
        match self {
            Delimiter::Parenthesis => token::Paren,
            Delimiter::Brace => token::Brace,
            Delimiter::Bracket => token::Bracket,
            Delimiter::None => token::NoDelim,
        }
    }
}

impl TokenTree {
    fn from_internal(stream: tokenstream::TokenStream, stack: &mut Vec<TokenTree>)
                -> TokenTree {
        use syntax::parse::token::*;

        let (tree, is_joint) = stream.as_tree();
        let (span, token) = match tree {
            tokenstream::TokenTree::Token(span, token) => (span, token),
            tokenstream::TokenTree::Delimited(span, delimed) => {
                let delimiter = Delimiter::from_internal(delimed.delim);
                let mut g = Group::new(delimiter, TokenStream(delimed.tts.into()));
                g.set_span(Span(span));
                return g.into()
            }
        };

        let op_kind = if is_joint { Spacing::Joint } else { Spacing::Alone };
        macro_rules! tt {
            ($e:expr) => ({
                let mut x = TokenTree::from($e);
                x.set_span(Span(span));
                x
            })
        }
        macro_rules! op {
            ($a:expr) => (tt!(Op::new($a, op_kind)));
            ($a:expr, $b:expr) => ({
                stack.push(tt!(Op::new($b, op_kind)));
                tt!(Op::new($a, Spacing::Joint))
            });
            ($a:expr, $b:expr, $c:expr) => ({
                stack.push(tt!(Op::new($c, op_kind)));
                stack.push(tt!(Op::new($b, Spacing::Joint)));
                tt!(Op::new($a, Spacing::Joint))
            })
        }

        match token {
            Eq => op!('='),
            Lt => op!('<'),
            Le => op!('<', '='),
            EqEq => op!('=', '='),
            Ne => op!('!', '='),
            Ge => op!('>', '='),
            Gt => op!('>'),
            AndAnd => op!('&', '&'),
            OrOr => op!('|', '|'),
            Not => op!('!'),
            Tilde => op!('~'),
            BinOp(Plus) => op!('+'),
            BinOp(Minus) => op!('-'),
            BinOp(Star) => op!('*'),
            BinOp(Slash) => op!('/'),
            BinOp(Percent) => op!('%'),
            BinOp(Caret) => op!('^'),
            BinOp(And) => op!('&'),
            BinOp(Or) => op!('|'),
            BinOp(Shl) => op!('<', '<'),
            BinOp(Shr) => op!('>', '>'),
            BinOpEq(Plus) => op!('+', '='),
            BinOpEq(Minus) => op!('-', '='),
            BinOpEq(Star) => op!('*', '='),
            BinOpEq(Slash) => op!('/', '='),
            BinOpEq(Percent) => op!('%', '='),
            BinOpEq(Caret) => op!('^', '='),
            BinOpEq(And) => op!('&', '='),
            BinOpEq(Or) => op!('|', '='),
            BinOpEq(Shl) => op!('<', '<', '='),
            BinOpEq(Shr) => op!('>', '>', '='),
            At => op!('@'),
            Dot => op!('.'),
            DotDot => op!('.', '.'),
            DotDotDot => op!('.', '.', '.'),
            DotDotEq => op!('.', '.', '='),
            Comma => op!(','),
            Semi => op!(';'),
            Colon => op!(':'),
            ModSep => op!(':', ':'),
            RArrow => op!('-', '>'),
            LArrow => op!('<', '-'),
            FatArrow => op!('=', '>'),
            Pound => op!('#'),
            Dollar => op!('$'),
            Question => op!('?'),

            Ident(ident, false) | Lifetime(ident) => {
                tt!(Term::new(&ident.name.as_str(), Span(span)))
            }
            Ident(ident, true) => {
                tt!(Term::new(&format!("r#{}", ident), Span(span)))
            }
            Literal(lit, suffix) => tt!(self::Literal { lit, suffix, span: Span(span) }),
            DocComment(c) => {
                let style = comments::doc_comment_style(&c.as_str());
                let stripped = comments::strip_doc_comment_decoration(&c.as_str());
                let stream = vec![
                    tt!(Term::new("doc", Span(span))),
                    tt!(Op::new('=', Spacing::Alone)),
                    tt!(self::Literal::string(&stripped)),
                ].into_iter().collect();
                stack.push(tt!(Group::new(Delimiter::Bracket, stream)));
                if style == ast::AttrStyle::Inner {
                    stack.push(tt!(Op::new('!', Spacing::Alone)));
                }
                tt!(Op::new('#', Spacing::Alone))
            }

            Interpolated(_) => {
                __internal::with_sess(|(sess, _)| {
                    let tts = token.interpolated_to_tokenstream(sess, span);
                    tt!(Group::new(Delimiter::None, TokenStream(tts)))
                })
            }

            DotEq => op!('.', '='),
            OpenDelim(..) | CloseDelim(..) => unreachable!(),
            Whitespace | Comment | Shebang(..) | Eof => unreachable!(),
        }
    }

    fn to_internal(self) -> tokenstream::TokenStream {
        use syntax::parse::token::*;
        use syntax::tokenstream::{TokenTree, Delimited};

        let (op, kind, span) = match self {
            self::TokenTree::Op(tt) => (tt.op(), tt.spacing(), tt.span()),
            self::TokenTree::Group(tt) => {
                return TokenTree::Delimited(tt.span.0, Delimited {
                    delim: tt.delimiter.to_internal(),
                    tts: tt.stream.0.into(),
                }).into();
            },
            self::TokenTree::Term(tt) => {
                let ident = ast::Ident::new(tt.sym, tt.span.0);
                let sym_str = tt.sym.as_str();
                let token = if sym_str.starts_with("'") {
                    Lifetime(ident)
                } else if sym_str.starts_with("r#") {
                    let name = Symbol::intern(&sym_str[2..]);
                    let ident = ast::Ident::new(name, ident.span);
                    Ident(ident, true)
                } else {
                    Ident(ident, false)
                };
                return TokenTree::Token(tt.span.0, token).into();
            }
            self::TokenTree::Literal(self::Literal {
                lit: Lit::Integer(ref a),
                suffix,
                span,
            })
                if a.as_str().starts_with("-") =>
            {
                let minus = BinOp(BinOpToken::Minus);
                let integer = Symbol::intern(&a.as_str()[1..]);
                let integer = Literal(Lit::Integer(integer), suffix);
                let a = TokenTree::Token(span.0, minus);
                let b = TokenTree::Token(span.0, integer);
                return vec![a, b].into_iter().collect()
            }
            self::TokenTree::Literal(self::Literal {
                lit: Lit::Float(ref a),
                suffix,
                span,
            })
                if a.as_str().starts_with("-") =>
            {
                let minus = BinOp(BinOpToken::Minus);
                let float = Symbol::intern(&a.as_str()[1..]);
                let float = Literal(Lit::Float(float), suffix);
                let a = TokenTree::Token(span.0, minus);
                let b = TokenTree::Token(span.0, float);
                return vec![a, b].into_iter().collect()
            }
            self::TokenTree::Literal(tt) => {
                let token = Literal(tt.lit, tt.suffix);
                return TokenTree::Token(tt.span.0, token).into()
            }
        };

        let token = match op {
            '=' => Eq,
            '<' => Lt,
            '>' => Gt,
            '!' => Not,
            '~' => Tilde,
            '+' => BinOp(Plus),
            '-' => BinOp(Minus),
            '*' => BinOp(Star),
            '/' => BinOp(Slash),
            '%' => BinOp(Percent),
            '^' => BinOp(Caret),
            '&' => BinOp(And),
            '|' => BinOp(Or),
            '@' => At,
            '.' => Dot,
            ',' => Comma,
            ';' => Semi,
            ':' => Colon,
            '#' => Pound,
            '$' => Dollar,
            '?' => Question,
            _ => panic!("unsupported character {}", op),
        };

        let tree = TokenTree::Token(span.0, token);
        match kind {
            Spacing::Alone => tree.into(),
            Spacing::Joint => tree.joint(),
        }
    }
}

/// Permanently unstable internal implementation details of this crate. This
/// should not be used.
///
/// These methods are used by the rest of the compiler to generate instances of
/// `TokenStream` to hand to macro definitions, as well as consume the output.
///
/// Note that this module is also intentionally separate from the rest of the
/// crate. This allows the `#[unstable]` directive below to naturally apply to
/// all of the contents.
#[unstable(feature = "proc_macro_internals", issue = "27812")]
#[doc(hidden)]
pub mod __internal {
    pub use quote::{LiteralKind, Quoter, unquote};

    use std::cell::Cell;

    use syntax::ast;
    use syntax::ext::base::ExtCtxt;
    use syntax::ext::hygiene::Mark;
    use syntax::ptr::P;
    use syntax::parse::{self, ParseSess};
    use syntax::parse::token::{self, Token};
    use syntax::tokenstream;
    use syntax_pos::{BytePos, Loc, DUMMY_SP};

    use super::{TokenStream, LexError};

    pub fn lookup_char_pos(pos: BytePos) -> Loc {
        with_sess(|(sess, _)| sess.codemap().lookup_char_pos(pos))
    }

    pub fn new_token_stream(item: P<ast::Item>) -> TokenStream {
        let token = Token::interpolated(token::NtItem(item));
        TokenStream(tokenstream::TokenTree::Token(DUMMY_SP, token).into())
    }

    pub fn token_stream_wrap(inner: tokenstream::TokenStream) -> TokenStream {
        TokenStream(inner)
    }

    pub fn token_stream_parse_items(stream: TokenStream) -> Result<Vec<P<ast::Item>>, LexError> {
        with_sess(move |(sess, _)| {
            let mut parser = parse::stream_to_parser(sess, stream.0);
            let mut items = Vec::new();

            while let Some(item) = try!(parser.parse_item().map_err(super::parse_to_lex_err)) {
                items.push(item)
            }

            Ok(items)
        })
    }

    pub fn token_stream_inner(stream: TokenStream) -> tokenstream::TokenStream {
        stream.0
    }

    pub trait Registry {
        fn register_custom_derive(&mut self,
                                  trait_name: &str,
                                  expand: fn(TokenStream) -> TokenStream,
                                  attributes: &[&'static str]);

        fn register_attr_proc_macro(&mut self,
                                    name: &str,
                                    expand: fn(TokenStream, TokenStream) -> TokenStream);

        fn register_bang_proc_macro(&mut self,
                                    name: &str,
                                    expand: fn(TokenStream) -> TokenStream);
    }

    // Emulate scoped_thread_local!() here essentially
    thread_local! {
        static CURRENT_SESS: Cell<(*const ParseSess, Mark)> =
            Cell::new((0 as *const _, Mark::root()));
    }

    pub fn set_sess<F, R>(cx: &ExtCtxt, f: F) -> R
        where F: FnOnce() -> R
    {
        struct Reset { prev: (*const ParseSess, Mark) }

        impl Drop for Reset {
            fn drop(&mut self) {
                CURRENT_SESS.with(|p| p.set(self.prev));
            }
        }

        CURRENT_SESS.with(|p| {
            let _reset = Reset { prev: p.get() };
            p.set((cx.parse_sess, cx.current_expansion.mark));
            f()
        })
    }

    pub fn in_sess() -> bool
    {
        let p = CURRENT_SESS.with(|p| p.get());
        !p.0.is_null()
    }

    pub fn with_sess<F, R>(f: F) -> R
        where F: FnOnce((&ParseSess, Mark)) -> R
    {
        let p = CURRENT_SESS.with(|p| p.get());
        assert!(!p.0.is_null(), "proc_macro::__internal::with_sess() called \
                                 before set_parse_sess()!");
        f(unsafe { (&*p.0, p.1) })
    }
}

fn parse_to_lex_err(mut err: DiagnosticBuilder) -> LexError {
    err.cancel();
    LexError { _inner: () }
}
