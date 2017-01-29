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
//! This file contains the implementation internals of the quasiquoter provided by `qquote!`.

use syntax::ast::Ident;
use syntax::parse::token::{self, Token, Lit};
use syntax::symbol::Symbol;
use syntax::tokenstream::{self, Delimited, TokenTree, TokenStream};
use syntax_pos::DUMMY_SP;

use std::rc::Rc;

pub fn qquote<'cx>(stream: TokenStream) -> TokenStream {
    stream.quote()
}

trait Quote {
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
    ($i:ident) => { Token::Ident(Ident::from_str(stringify!($i))) };
}

macro_rules! quote_tree {
    ((unquote $($t:tt)*)) => { $($t)* };
    ((quote $($t:tt)*)) => { ($($t)*).quote() };
    (($($t:tt)*)) => { delimit(token::Paren, quote!($($t)*)) };
    ([$($t:tt)*]) => { delimit(token::Bracket, quote!($($t)*)) };
    ({$($t:tt)*}) => { delimit(token::Brace, quote!($($t)*)) };
    ($t:tt) => { TokenStream::from(TokenTree::Token(DUMMY_SP, quote_tok!($t))) };
}

fn delimit(delim: token::DelimToken, stream: TokenStream) -> TokenStream {
    TokenTree::Delimited(DUMMY_SP, Rc::new(Delimited {
        delim: delim,
        tts: stream.trees().cloned().collect(),
    })).into()
}

macro_rules! quote {
    () => { TokenStream::empty() };
    ($($t:tt)*) => { [ $( quote_tree!($t), )* ].iter().cloned().collect::<TokenStream>() };
}

impl<T: Quote> Quote for Option<T> {
    fn quote(&self) -> TokenStream {
        match *self {
            Some(ref t) => quote!(::std::option::Option::Some((quote t))),
            None => quote!(::std::option::Option::None),
        }
    }
}

impl Quote for TokenStream {
    fn quote(&self) -> TokenStream {
        if self.is_empty() {
            return quote!(::syntax::tokenstream::TokenStream::empty());
        }

        struct Quote<'a>(tokenstream::Cursor<'a>);

        impl<'a> Iterator for Quote<'a> {
            type Item = TokenStream;

            fn next(&mut self) -> Option<TokenStream> {
                let is_unquote = match self.0.peek() {
                    Some(&TokenTree::Token(_, Token::Ident(ident))) if ident.name == "unquote" => {
                        self.0.next();
                        true
                    }
                    _ => false,
                };

                self.0.next().cloned().map(|tree| {
                    let quoted_tree = if is_unquote { tree.into() } else { tree.quote() };
                    quote!(::syntax::tokenstream::TokenStream::from((unquote quoted_tree)),)
                })
            }
        }

        let quoted = Quote(self.trees()).collect::<TokenStream>();
        quote!([(unquote quoted)].iter().cloned().collect::<::syntax::tokenstream::TokenStream>())
    }
}

impl Quote for Vec<TokenTree> {
    fn quote(&self) -> TokenStream {
        let stream = self.iter().cloned().collect::<TokenStream>();
        quote!((quote stream).trees().cloned().collect::<::std::vec::Vec<_> >())
    }
}

impl Quote for TokenTree {
    fn quote(&self) -> TokenStream {
        match *self {
            TokenTree::Token(_, ref token) => quote! {
                ::syntax::tokenstream::TokenTree::Token(::syntax::ext::quote::rt::DUMMY_SP,
                                                        (quote token))
            },
            TokenTree::Delimited(_, ref delimited) => quote! {
                ::syntax::tokenstream::TokenTree::Delimited(::syntax::ext::quote::rt::DUMMY_SP,
                                                            (quote delimited))
            },
        }
    }
}

impl Quote for Rc<Delimited> {
    fn quote(&self) -> TokenStream {
        quote!(::std::rc::Rc::new(::syntax::tokenstream::Delimited {
            delim: (quote self.delim),
            tts: (quote self.tts),
        }))
    }
}

impl<'a> Quote for &'a str {
    fn quote(&self) -> TokenStream {
        TokenTree::Token(DUMMY_SP, Token::Literal(token::Lit::Str_(Symbol::intern(self)), None))
            .into()
    }
}

impl Quote for Ident {
    fn quote(&self) -> TokenStream {
        // FIXME(jseyfried) quote hygiene
        quote!(::syntax::ast::Ident::from_str((quote &*self.name.as_str())))
    }
}

impl Quote for Symbol {
    fn quote(&self) -> TokenStream {
        quote!(::syntax::symbol::Symbol::intern((quote &*self.as_str())))
    }
}

impl Quote for Token {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*; $($t:tt)*) => {
                match *self {
                    $( Token::$i => quote!(::syntax::parse::token::$i), )*
                    $( $t )*
                }
            }
        }

        gen_match! {
            Eq, Lt, Le, EqEq, Ne, Ge, Gt, AndAnd, OrOr, Not, Tilde, At, Dot, DotDot, DotDotDot,
            Comma, Semi, Colon, ModSep, RArrow, LArrow, FatArrow, Pound, Dollar, Question,
            Underscore;

            Token::OpenDelim(delim) => quote!(::syntax::parse::token::OpenDelim((quote delim))),
            Token::CloseDelim(delim) => quote!(::syntax::parse::token::CloseDelim((quote delim))),
            Token::BinOp(tok) => quote!(::syntax::parse::token::BinOp((quote tok))),
            Token::BinOpEq(tok) => quote!(::syntax::parse::token::BinOpEq((quote tok))),
            Token::Ident(ident) => quote!(::syntax::parse::token::Ident((quote ident))),
            Token::Lifetime(ident) => quote!(::syntax::parse::token::Lifetime((quote ident))),
            Token::Literal(lit, sfx) => quote! {
                ::syntax::parse::token::Literal((quote lit), (quote sfx))
            },
            _ => panic!("Unhandled case!"),
        }
    }
}

impl Quote for token::BinOpToken {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*) => {
                match *self {
                    $( token::BinOpToken::$i => quote!(::syntax::parse::token::BinOpToken::$i), )*
                }
            }
        }

        gen_match!(Plus, Minus, Star, Slash, Percent, Caret, And, Or, Shl, Shr)
    }
}

impl Quote for Lit {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*) => {
                match *self {
                    $( Lit::$i(lit) => quote!(::syntax::parse::token::Lit::$i((quote lit))), )*
                    _ => panic!("Unsupported literal"),
                }
            }
        }

        gen_match!(Byte, Char, Float, Str_, Integer, ByteStr)
    }
}

impl Quote for token::DelimToken {
    fn quote(&self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident),*) => {
                match *self {
                    $(token::DelimToken::$i => { quote!(::syntax::parse::token::DelimToken::$i) })*
                }
            }
        }

        gen_match!(Paren, Bracket, Brace, NoDelim)
    }
}
