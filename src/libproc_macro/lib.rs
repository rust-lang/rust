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

#![crate_name = "proc_macro"]
#![stable(feature = "proc_macro_lib", since = "1.15.0")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![deny(warnings)]
#![deny(missing_docs)]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables, unused_mut))))]

#![feature(i128_type)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(lang_items)]

extern crate syntax;
extern crate syntax_pos;

use std::{fmt, iter, ops};
use std::str::FromStr;

use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::parse::{self, token};
use syntax::symbol;
use syntax::tokenstream;
use syntax_pos::DUMMY_SP;
use syntax_pos::SyntaxContext;

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
#[derive(Debug)]
#[stable(feature = "proc_macro_lib", since = "1.15.0")]
pub struct LexError {
    _inner: (),
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        __internal::with_sess(|(sess, mark)| {
            let src = src.to_string();
            let name = "<proc-macro source code>".to_string();
            let call_site = mark.expn_info().unwrap().call_site;
            let stream = parse::parse_stream_from_source_str(name, src, sess, Some(call_site));
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

/// `quote!(..)` accepts arbitrary tokens and expands into a `TokenStream` describing the input.
/// For example, `quote!(a + b)` will produce a expression, that, when evaluated, constructs
/// constructs the `TokenStream` `[Word("a"), Op('+', Alone), Word("b")]`.
///
/// Unquoting is done with `$`, and works by taking the single next ident as the unquoted term.
/// To quote `$` itself, use `$$`.
#[unstable(feature = "proc_macro", issue = "38356")]
#[macro_export]
macro_rules! quote { () => {} }

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream(tree.to_raw())
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<TokenKind> for TokenStream {
    fn from(kind: TokenKind) -> TokenStream {
        TokenTree::from(kind).into()
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl<T: Into<TokenStream>> iter::FromIterator<T> for TokenStream {
    fn from_iter<I: IntoIterator<Item = T>>(streams: I) -> Self {
        let mut builder = tokenstream::TokenStream::builder();
        for stream in streams {
            builder.push(stream.into().0);
        }
        TokenStream(builder.build())
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl IntoIterator for TokenStream {
    type Item = TokenTree;
    type IntoIter = TokenIter;

    fn into_iter(self) -> TokenIter {
        TokenIter { cursor: self.0.trees(), next: None }
    }
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

/// A region of source code, along with macro expansion information.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Copy, Clone)]
pub struct Span(syntax_pos::Span);

#[unstable(feature = "proc_macro", issue = "38356")]
impl Default for Span {
    fn default() -> Span {
        ::__internal::with_sess(|(_, mark)| Span(syntax_pos::Span {
            ctxt: SyntaxContext::empty().apply_mark(mark),
            ..mark.expn_info().unwrap().call_site
        }))
    }
}

impl Span {
    /// The span of the invocation of the current procedural macro.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn call_site() -> Span {
        ::__internal::with_sess(|(_, mark)| Span(mark.expn_info().unwrap().call_site))
    }
}

/// A single token or a delimited sequence of token trees (e.g. `[1, (), ..]`).
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Clone)]
pub struct TokenTree {
    /// The `TokenTree`'s span
    pub span: Span,
    /// Description of the `TokenTree`
    pub kind: TokenKind,
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl From<TokenKind> for TokenTree {
    fn from(kind: TokenKind) -> TokenTree {
        TokenTree { span: Span::default(), kind: kind }
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        TokenStream::from(self.clone()).fmt(f)
    }
}

/// Description of a `TokenTree`
#[derive(Clone)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub enum TokenKind {
    /// A delimited tokenstream.
    Sequence(Delimiter, TokenStream),
    /// A unicode identifier.
    Word(Symbol),
    /// A punctuation character (`+`, `,`, `$`, etc.).
    Op(char, OpKind),
    /// A literal character (`'a'`), string (`"hello"`), or number (`2.3`).
    Literal(Literal),
}

/// Describes how a sequence of token trees is delimited.
#[derive(Copy, Clone)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `[ ... ]`
    Brace,
    /// `{ ... }`
    Bracket,
    /// An implicit delimiter, e.g. `$var`, where $var is  `...`.
    None,
}

/// An interned string.
#[derive(Copy, Clone)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct Symbol(symbol::Symbol);

#[unstable(feature = "proc_macro", issue = "38356")]
impl<'a> From<&'a str> for Symbol {
    fn from(string: &'a str) -> Symbol {
        Symbol(symbol::Symbol::intern(string))
    }
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl ops::Deref for Symbol {
    type Target = str;

    fn deref(&self) -> &str {
        unsafe { &*(self.0.as_str().deref() as *const str) }
    }
}

/// Whether an `Op` is either followed immediately by another `Op` or followed by whitespace.
#[derive(Copy, Clone)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub enum OpKind {
    /// e.g. `+` is `Alone` in `+ =`.
    Alone,
    /// e.g. `+` is `Joint` in `+=`.
    Joint,
}

/// A literal character (`'a'`), string (`"hello"`), or number (`2.3`).
#[derive(Clone)]
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct Literal(token::Token);

#[unstable(feature = "proc_macro", issue = "38356")]
impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        TokenTree { kind: TokenKind::Literal(self.clone()), span: Span(DUMMY_SP) }.fmt(f)
    }
}

macro_rules! int_literals {
    ($($int_kind:ident),*) => {$(
        /// Integer literal.
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $int_kind(n: $int_kind) -> Literal {
            Literal::integer(n as i128, stringify!($int_kind))
        }
    )*}
}

impl Literal {
    int_literals!(u8, i8, u16, i16, u32, i32, u64, i64);
    fn integer(n: i128, kind: &'static str) -> Literal {
        Literal(token::Literal(token::Lit::Integer(symbol::Symbol::intern(&n.to_string())),
                               Some(symbol::Symbol::intern(kind))))
    }

    /// Floating point literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f32(n: f32) -> Literal {
        Literal(token::Literal(token::Lit::Float(symbol::Symbol::intern(&n.to_string())),
                               Some(symbol::Symbol::intern("f32"))))
    }

    /// Floating point literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn f64(n: f32) -> Literal {
        Literal(token::Literal(token::Lit::Float(symbol::Symbol::intern(&n.to_string())),
                               Some(symbol::Symbol::intern("f64"))))
    }

    /// String literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn string(string: &str) -> Literal {
        let mut escaped = String::new();
        for ch in string.chars() {
            escaped.extend(ch.escape_unicode());
        }
        Literal(token::Literal(token::Lit::Str_(symbol::Symbol::intern(&escaped)), None))
    }

    /// Character literal.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn character(ch: char) -> Literal {
        let mut escaped = String::new();
        escaped.extend(ch.escape_unicode());
        Literal(token::Literal(token::Lit::Char(symbol::Symbol::intern(&escaped)), None))
    }
}

/// An iterator over `TokenTree`s.
#[unstable(feature = "proc_macro", issue = "38356")]
pub struct TokenIter {
    cursor: tokenstream::Cursor,
    next: Option<tokenstream::TokenStream>,
}

#[unstable(feature = "proc_macro", issue = "38356")]
impl Iterator for TokenIter {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.next.take().or_else(|| self.cursor.next_as_stream())
            .map(|next| TokenTree::from_raw(next, &mut self.next))
    }
}

impl Delimiter {
    fn from_raw(delim: token::DelimToken) -> Delimiter {
        match delim {
            token::Paren => Delimiter::Parenthesis,
            token::Brace => Delimiter::Brace,
            token::Bracket => Delimiter::Bracket,
            token::NoDelim => Delimiter::None,
        }
    }

    fn to_raw(self) -> token::DelimToken {
        match self {
            Delimiter::Parenthesis => token::Paren,
            Delimiter::Brace => token::Brace,
            Delimiter::Bracket => token::Bracket,
            Delimiter::None => token::NoDelim,
        }
    }
}

impl TokenTree {
    fn from_raw(stream: tokenstream::TokenStream, next: &mut Option<tokenstream::TokenStream>)
                -> TokenTree {
        use syntax::parse::token::*;

        let (tree, is_joint) = stream.as_tree();
        let (mut span, token) = match tree {
            tokenstream::TokenTree::Token(span, token) => (span, token),
            tokenstream::TokenTree::Delimited(span, delimed) => {
                let delimiter = Delimiter::from_raw(delimed.delim);
                return TokenTree {
                    span: Span(span),
                    kind: TokenKind::Sequence(delimiter, TokenStream(delimed.tts.into())),
                };
            }
        };

        let op_kind = if is_joint { OpKind::Joint } else { OpKind::Alone };
        macro_rules! op {
            ($op:expr) => { TokenKind::Op($op, op_kind) }
        }

        macro_rules! joint {
            ($first:expr, $rest:expr) => { joint($first, $rest, is_joint, &mut span, next) }
        }

        fn joint(first: char, rest: Token, is_joint: bool, span: &mut syntax_pos::Span,
                 next: &mut Option<tokenstream::TokenStream>)
                 -> TokenKind {
            let (first_span, rest_span) = (*span, *span);
            *span = first_span;
            let tree = tokenstream::TokenTree::Token(rest_span, rest);
            *next = Some(if is_joint { tree.joint() } else { tree.into() });
            TokenKind::Op(first, OpKind::Joint)
        }

        let kind = match token {
            Eq => op!('='),
            Lt => op!('<'),
            Le => joint!('<', Eq),
            EqEq => joint!('=', Eq),
            Ne => joint!('!', Eq),
            Ge => joint!('>', Eq),
            Gt => op!('>'),
            AndAnd => joint!('&', BinOp(And)),
            OrOr => joint!('|', BinOp(Or)),
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
            BinOp(Shl) => joint!('<', Lt),
            BinOp(Shr) => joint!('>', Gt),
            BinOpEq(Plus) => joint!('+', Eq),
            BinOpEq(Minus) => joint!('-', Eq),
            BinOpEq(Star) => joint!('*', Eq),
            BinOpEq(Slash) => joint!('/', Eq),
            BinOpEq(Percent) => joint!('%', Eq),
            BinOpEq(Caret) => joint!('^', Eq),
            BinOpEq(And) => joint!('&', Eq),
            BinOpEq(Or) => joint!('|', Eq),
            BinOpEq(Shl) => joint!('<', Le),
            BinOpEq(Shr) => joint!('>', Ge),
            At => op!('@'),
            Dot => op!('.'),
            DotDot => joint!('.', Dot),
            DotDotDot => joint!('.', DotDot),
            Comma => op!(','),
            Semi => op!(';'),
            Colon => op!(':'),
            ModSep => joint!(':', Colon),
            RArrow => joint!('-', Gt),
            LArrow => joint!('<', BinOp(Minus)),
            FatArrow => joint!('=', Gt),
            Pound => op!('#'),
            Dollar => op!('$'),
            Question => op!('?'),
            Underscore => op!('_'),

            Ident(ident) | Lifetime(ident) => TokenKind::Word(Symbol(ident.name)),
            Literal(..) | DocComment(..) => TokenKind::Literal(self::Literal(token)),

            Interpolated(..) => unimplemented!(),

            OpenDelim(..) | CloseDelim(..) => unreachable!(),
            Whitespace | Comment | Shebang(..) | Eof => unreachable!(),
        };

        TokenTree { span: Span(span), kind: kind }
    }

    fn to_raw(self) -> tokenstream::TokenStream {
        use syntax::parse::token::*;
        use syntax::tokenstream::{TokenTree, Delimited};

        let (op, kind) = match self.kind {
            TokenKind::Op(op, kind) => (op, kind),
            TokenKind::Sequence(delimiter, tokens) => {
                return TokenTree::Delimited(self.span.0, Delimited {
                    delim: delimiter.to_raw(),
                    tts: tokens.0.into(),
                }).into();
            },
            TokenKind::Word(symbol) => {
                let ident = ast::Ident { name: symbol.0, ctxt: self.span.0.ctxt };
                let token =
                    if symbol.0.as_str().starts_with("'") { Lifetime(ident) } else { Ident(ident) };
                return TokenTree::Token(self.span.0, token).into();
            }
            TokenKind::Literal(token) => return TokenTree::Token(self.span.0, token.0).into(),
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
            '_' => Underscore,
            _ => panic!("unsupported character {}", op),
        };

        let tree = TokenTree::Token(self.span.0, token);
        match kind {
            OpKind::Alone => tree.into(),
            OpKind::Joint => tree.joint(),
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
#[path = ""]
pub mod __internal {
    mod quote;
    pub use self::quote::{Quoter, __rt};

    use std::cell::Cell;
    use std::rc::Rc;

    use syntax::ast;
    use syntax::ext::base::ExtCtxt;
    use syntax::ext::hygiene::Mark;
    use syntax::ptr::P;
    use syntax::parse::{self, token, ParseSess};
    use syntax::tokenstream;

    use super::{TokenStream, LexError};

    pub fn new_token_stream(item: P<ast::Item>) -> TokenStream {
        let (span, token) = (item.span, token::Interpolated(Rc::new(token::NtItem(item))));
        TokenStream(tokenstream::TokenTree::Token(span, token).into())
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
