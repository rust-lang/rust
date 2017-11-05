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

use {Delimiter, Literal, Spacing, Span, Term, TokenNode, TokenStream, TokenTree};

use std::iter;
use syntax::ext::base::{ExtCtxt, ProcMacro};
use syntax::parse::token;
use syntax::tokenstream;

pub struct Quoter;

pub mod __rt {
    pub use syntax::ast::Ident;
    pub use syntax::parse::token;
    pub use syntax::symbol::Symbol;
    pub use syntax::tokenstream::{TokenStream, TokenStreamBuilder, TokenTree, Delimited};

    use syntax_pos::Span;
    use syntax_pos::hygiene::SyntaxContext;

    pub fn unquote<T: Into<::TokenStream> + Clone>(tokens: &T) -> TokenStream {
        T::into(tokens.clone()).0
    }

    pub fn ctxt() -> SyntaxContext {
        ::__internal::with_sess(|(_, mark)| SyntaxContext::empty().apply_mark(mark))
    }

    pub fn span() -> Span {
        ::Span::default().0
    }
}

pub trait Quote {
    fn quote(self) -> TokenStream;
}

macro_rules! quote_tok {
    (,) => { TokenNode::Op(',', Spacing::Alone) };
    (.) => { TokenNode::Op('.', Spacing::Alone) };
    (:) => { TokenNode::Op(':', Spacing::Alone) };
    (::) => {
        [
            TokenNode::Op(':', Spacing::Joint),
            TokenNode::Op(':', Spacing::Alone)
        ].iter().cloned().collect::<TokenStream>()
    };
    (!) => { TokenNode::Op('!', Spacing::Alone) };
    (<) => { TokenNode::Op('<', Spacing::Alone) };
    (>) => { TokenNode::Op('>', Spacing::Alone) };
    (_) => { TokenNode::Op('_', Spacing::Alone) };
    (0) => { TokenNode::Literal(::Literal::integer(0)) };
    (&) => { TokenNode::Op('&', Spacing::Alone) };
    ($i:ident) => { TokenNode::Term(Term::intern(stringify!($i))) };
}

macro_rules! quote_tree {
    ((unquote $($t:tt)*)) => { $($t)* };
    ((quote $($t:tt)*)) => { ($($t)*).quote() };
    (($($t:tt)*)) => { TokenNode::Group(Delimiter::Parenthesis, quote!($($t)*)) };
    ([$($t:tt)*]) => { TokenNode::Group(Delimiter::Bracket, quote!($($t)*)) };
    ({$($t:tt)*}) => { TokenNode::Group(Delimiter::Brace, quote!($($t)*)) };
    (rt) => { quote!(::__internal::__rt) };
    ($t:tt) => { quote_tok!($t) };
}

macro_rules! quote {
    () => { TokenStream::empty() };
    ($($t:tt)*) => {
        [
            $(TokenStream::from(quote_tree!($t)),)*
        ].iter().cloned().collect::<TokenStream>()
    };
}

impl ProcMacro for Quoter {
    fn expand<'cx>(&self, cx: &'cx mut ExtCtxt,
                   _: ::syntax_pos::Span,
                   stream: tokenstream::TokenStream)
                   -> tokenstream::TokenStream {
        let mut info = cx.current_expansion.mark.expn_info().unwrap();
        info.callee.allow_internal_unstable = true;
        cx.current_expansion.mark.set_expn_info(info);
        ::__internal::set_sess(cx, || quote!(::TokenStream {
            0: (quote TokenStream(stream))
        }).0)
    }
}

impl<T: Quote> Quote for Option<T> {
    fn quote(self) -> TokenStream {
        match self {
            Some(t) => quote!(Some((quote t))),
            None => quote!(None),
        }
    }
}

impl Quote for TokenStream {
    fn quote(self) -> TokenStream {
        let mut after_dollar = false;
        let stream = iter::once(quote!(rt::TokenStreamBuilder::new()))
            .chain(self.into_iter().filter_map(|tree| {
                if after_dollar {
                    after_dollar = false;
                    match tree.kind {
                        TokenNode::Term(_) => {
                            return Some(quote!(.add(rt::unquote(&(unquote tree)))));
                        }
                        TokenNode::Op('$', _) => {}
                        _ => panic!("`$` must be followed by an ident or `$` in `quote!`"),
                    }
                } else if let TokenNode::Op('$', _) = tree.kind {
                    after_dollar = true;
                    return None;
                }

                Some(quote!(.add(rt::TokenStream::from((quote tree)))))
            }))
            .chain(iter::once(quote!(.build()))).collect();

        if after_dollar {
            panic!("unexpected trailing `$` in `quote!`");
        }

        stream
    }
}

impl Quote for TokenTree {
    fn quote(self) -> TokenStream {
        let (op, kind) = match self.kind {
            TokenNode::Op(op, kind) => (op, kind),
            TokenNode::Group(delimiter, tokens) => {
                return quote! {
                    rt::TokenTree::Delimited((quote self.span), rt::Delimited {
                        delim: (quote delimiter),
                        tts: (quote tokens).into()
                    })
                };
            },
            TokenNode::Term(term) => {
                let variant = if term.as_str().starts_with("'") {
                    quote!(Lifetime)
                } else {
                    quote!(Ident)
                };
                return quote! {
                    rt::TokenTree::Token((quote self.span),
                        rt::token::(unquote variant)(rt::Ident {
                            name: (quote term),
                            ctxt: rt::ctxt()
                        }))
                };
            }
            TokenNode::Literal(lit) => {
                return quote! {
                    rt::TokenTree::Token((quote self.span), (quote lit))
                };
            }
        };

        let token = match op {
            '=' => quote!(Eq),
            '<' => quote!(Lt),
            '>' => quote!(Gt),
            '!' => quote!(Not),
            '~' => quote!(Tilde),
            '+' => quote!(BinOp(rt::token::BinOpToken::Plus)),
            '-' => quote!(BinOp(rt::token::BinOpToken::Minus)),
            '*' => quote!(BinOp(rt::token::BinOpToken::Star)),
            '/' => quote!(BinOp(rt::token::BinOpToken::Slash)),
            '%' => quote!(BinOp(rt::token::BinOpToken::Percent)),
            '^' => quote!(BinOp(rt::token::BinOpToken::Caret)),
            '&' => quote!(BinOp(rt::token::BinOpToken::And)),
            '|' => quote!(BinOp(rt::token::BinOpToken::Or)),
            '@' => quote!(At),
            '.' => quote!(Dot),
            ',' => quote!(Comma),
            ';' => quote!(Semi),
            ':' => quote!(Colon),
            '#' => quote!(Pound),
            '$' => quote!(Dollar),
            '?' => quote!(Question),
            '_' => quote!(Underscore),
            _ => panic!("unsupported character {}", op),
        };

        match kind {
            Spacing::Alone => quote! {
                rt::TokenTree::Token((quote self.span), rt::token::(unquote token))
            },
            Spacing::Joint => quote! {
                rt::TokenTree::Token((quote self.span), rt::token::(unquote token)).joint()
            },
        }
    }
}

impl<'a> Quote for &'a str {
    fn quote(self) -> TokenStream {
        TokenNode::Literal(Literal::string(self)).into()
    }
}

impl Quote for usize {
    fn quote(self) -> TokenStream {
        TokenNode::Literal(Literal::integer(self as i128)).into()
    }
}

impl Quote for Term {
    fn quote(self) -> TokenStream {
        quote!(rt::Symbol::intern((quote self.as_str())))
    }
}

impl Quote for Span {
    fn quote(self) -> TokenStream {
        quote!(rt::span())
    }
}

impl Quote for Literal {
    fn quote(self) -> TokenStream {
        let (lit, sfx) = match self.0 {
            token::Literal(lit, sfx) => (lit, sfx.map(Term)),
            _ => panic!("unsupported literal {:?}", self.0),
        };

        macro_rules! gen_match {
            ($($i:ident),*; $($raw:ident),*) => {
                match lit {
                    $(token::Lit::$i(lit) => quote! {
                        rt::token::Literal(rt::token::Lit::$i((quote Term(lit))),
                            (quote sfx))
                    },)*
                    $(token::Lit::$raw(lit, n) => quote! {
                        rt::token::Literal(rt::token::Lit::$raw((quote Term(lit)), (quote n)),
                            (quote sfx))
                    },)*
                }
            }
        }

        gen_match!(Byte, Char, Float, Str_, Integer, ByteStr; StrRaw, ByteStrRaw)
    }
}

impl Quote for Delimiter {
    fn quote(self) -> TokenStream {
        macro_rules! gen_match {
            ($($i:ident => $j:ident),*) => {
                match self {
                    $(Delimiter::$i => { quote!(rt::token::DelimToken::$j) })*
                }
            }
        }

        gen_match!(Parenthesis => Paren, Brace => Brace, Bracket => Bracket, None => NoDelim)
    }
}
