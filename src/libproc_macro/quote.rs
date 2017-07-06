// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably use items from `__rt`,
//! including re-exported API `libsyntax`, to build a `syntax::tokenstream::TokenStream`
//! and wrap it into a `proc_macro::TokenStream`.

use syntax::ast::Ident;
use syntax::ext::base::{ExtCtxt, ProcMacro};
use syntax::parse::token::{self, Token, Lit};
use syntax::symbol::Symbol;
use syntax::tokenstream::{Delimited, TokenTree, TokenStream, TokenStreamBuilder};
use syntax_pos::{DUMMY_SP, Span};
use syntax_pos::hygiene::SyntaxContext;

pub struct Quoter;

pub mod __rt {
    pub use syntax::ast::Ident;
    pub use syntax::parse::token;
    pub use syntax::symbol::Symbol;
    pub use syntax::tokenstream::{TokenStream, TokenStreamBuilder, TokenTree, Delimited};
    pub use super::{ctxt, span};

    pub fn unquote<T: Into<::TokenStream> + Clone>(tokens: &T) -> TokenStream {
        T::into(tokens.clone()).0
    }
}

pub fn ctxt() -> SyntaxContext {
    ::__internal::with_sess(|(_, mark)| SyntaxContext::empty().apply_mark(mark))
}

pub fn span() -> Span {
    ::Span::default().0
}

pub trait Quote {
    fn quote(&self) -> TokenStream;
}

macro_rules! quote_tok {
    (,) => { Token::Comma };
    (.) => { Token::Dot };
    (:) => { Token::Colon };
    (::) => { Token::ModSep };
    (!) => { Token::Not };
    (<) => { Token::Lt };
    (>) => { Token::Gt };
    (_) => { Token::Underscore };
    (0) => { Token::Literal(token::Lit::Integer(Symbol::intern("0")), None) };
    (&) => { Token::BinOp(token::And) };
    ($i:ident) => { Token::Ident(Ident { name: Symbol::intern(stringify!($i)), ctxt: ctxt() }) };
}

macro_rules! quote_tree {
    ((unquote $($t:tt)*)) => { TokenStream::from($($t)*) };
    ((quote $($t:tt)*)) => { ($($t)*).quote() };
    (($($t:tt)*)) => { delimit(token::Paren, quote!($($t)*)) };
    ([$($t:tt)*]) => { delimit(token::Bracket, quote!($($t)*)) };
    ({$($t:tt)*}) => { delimit(token::Brace, quote!($($t)*)) };
    (rt) => { quote!(::__internal::__rt) };
    ($t:tt) => { TokenStream::from(TokenTree::Token(span(), quote_tok!($t))) };
}

fn delimit(delim: token::DelimToken, stream: TokenStream) -> TokenStream {
    TokenTree::Delimited(span(), Delimited { delim: delim, tts: stream.into() }).into()
}

macro_rules! quote {
    () => { TokenStream::empty() };
    ($($t:tt)*) => { [ $( quote_tree!($t), )* ].iter().cloned().collect::<TokenStream>() };
}

impl ProcMacro for Quoter {
    fn expand<'cx>(&self, cx: &'cx mut ExtCtxt, _: Span, stream: TokenStream) -> TokenStream {
        let mut info = cx.current_expansion.mark.expn_info().unwrap();
        info.callee.allow_internal_unstable = true;
        cx.current_expansion.mark.set_expn_info(info);
        ::__internal::set_sess(cx, || quote!(::TokenStream((quote stream))))
    }
}

impl<T: Quote> Quote for Option<T> {
    fn quote(&self) -> TokenStream {
        match *self {
            Some(ref t) => quote!(Some((quote t))),
            None => quote!(None),
        }
    }
}

impl Quote for TokenStream {
    fn quote(&self) -> TokenStream {
        let mut builder = TokenStreamBuilder::new();
        builder.push(quote!(rt::TokenStreamBuilder::new()));

        let mut trees = self.trees();
        loop {
            let (mut tree, mut is_joint) = match trees.next_as_stream() {
                Some(next) => next.as_tree(),
                None => return builder.add(quote!(.build())).build(),
            };
            if let TokenTree::Token(_, Token::Dollar) = tree {
                let (next_tree, next_is_joint) = match trees.next_as_stream() {
                    Some(next) => next.as_tree(),
                    None => panic!("unexpected trailing `$` in `quote!`"),
                };
                match next_tree {
                    TokenTree::Token(_, Token::Ident(..)) => {
                        builder.push(quote!(.add(rt::unquote(&(unquote next_tree)))));
                        continue
                    }
                    TokenTree::Token(_, Token::Dollar) => {
                        tree = next_tree;
                        is_joint = next_is_joint;
                    }
                    _ => panic!("`$` must be followed by an ident or `$` in `quote!`"),
                }
            }

            builder.push(match is_joint {
                true => quote!(.add((quote tree).joint())),
                false => quote!(.add(rt::TokenStream::from((quote tree)))),
            });
        }
    }
}

impl Quote for TokenTree {
    fn quote(&self) -> TokenStream {
        match *self {
            TokenTree::Token(span, ref token) => quote! {
                rt::TokenTree::Token((quote span), (quote token))
            },
            TokenTree::Delimited(span, ref delimited) => quote! {
                rt::TokenTree::Delimited((quote span), (quote delimited))
            },
        }
    }
}

impl Quote for Delimited {
    fn quote(&self) -> TokenStream {
        quote!(rt::Delimited { delim: (quote self.delim), tts: (quote self.stream()).into() })
    }
}

impl<'a> Quote for &'a str {
    fn quote(&self) -> TokenStream {
        TokenTree::Token(span(), Token::Literal(token::Lit::Str_(Symbol::intern(self)), None))
            .into()
    }
}

impl Quote for usize {
    fn quote(&self) -> TokenStream {
        let integer_symbol = Symbol::intern(&self.to_string());
        TokenTree::Token(DUMMY_SP, Token::Literal(token::Lit::Integer(integer_symbol), None))
            .into()
    }
}

impl Quote for Ident {
    fn quote(&self) -> TokenStream {
        quote!(rt::Ident { name: (quote self.name), ctxt: rt::ctxt() })
    }
}

impl Quote for Symbol {
    fn quote(&self) -> TokenStream {
        quote!(rt::Symbol::intern((quote &*self.as_str())))
    }
}

impl Quote for Span {
    fn quote(&self) -> TokenStream {
        quote!(rt::span())
    }
}

impl Quote for Token {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*; $($t:tt)*) => {
                match *self {
                    $( Token::$i => quote!(rt::token::$i), )*
                    $( $t )*
                }
            }
        }

        gen_match! {
            Eq, Lt, Le, EqEq, Ne, Ge, Gt, AndAnd, OrOr, Not, Tilde, At, Dot, DotDot, DotDotDot,
            Comma, Semi, Colon, ModSep, RArrow, LArrow, FatArrow, Pound, Dollar, Question,
            Underscore;

            Token::OpenDelim(delim) => quote!(rt::token::OpenDelim((quote delim))),
            Token::CloseDelim(delim) => quote!(rt::token::CloseDelim((quote delim))),
            Token::BinOp(tok) => quote!(rt::token::BinOp((quote tok))),
            Token::BinOpEq(tok) => quote!(rt::token::BinOpEq((quote tok))),
            Token::Ident(ident) => quote!(rt::token::Ident((quote ident))),
            Token::Lifetime(ident) => quote!(rt::token::Lifetime((quote ident))),
            Token::Literal(lit, sfx) => quote!(rt::token::Literal((quote lit), (quote sfx))),
            _ => panic!("Unhandled case!"),
        }
    }
}

impl Quote for token::BinOpToken {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*) => {
                match *self {
                    $( token::BinOpToken::$i => quote!(rt::token::BinOpToken::$i), )*
                }
            }
        }

        gen_match!(Plus, Minus, Star, Slash, Percent, Caret, And, Or, Shl, Shr)
    }
}

impl Quote for Lit {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*; $($raw:ident),*) => {
                match *self {
                    $( Lit::$i(lit) => quote!(rt::token::Lit::$i((quote lit))), )*
                    $( Lit::$raw(lit, n) => {
                        quote!(::syntax::parse::token::Lit::$raw((quote lit), (quote n)))
                    })*
                }
            }
        }

        gen_match!(Byte, Char, Float, Str_, Integer, ByteStr; StrRaw, ByteStrRaw)
    }
}

impl Quote for token::DelimToken {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*) => {
                match *self {
                    $(token::DelimToken::$i => { quote!(rt::token::DelimToken::$i) })*
                }
            }
        }

        gen_match!(Paren, Bracket, Brace, NoDelim)
    }
}
