// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Delimiter, Level, Spacing, Span, __internal};
use {Group, Ident, Literal, Punct, TokenTree};

use rustc_errors as errors;
use syntax::ast;
use syntax::parse::lexer::comments;
use syntax::parse::token;
use syntax::tokenstream;
use syntax_pos::symbol::{keywords, Symbol};

impl Ident {
    pub(crate) fn new_maybe_raw(string: &str, span: Span, is_raw: bool) -> Ident {
        let sym = Symbol::intern(string);
        if is_raw
            && (sym == keywords::Underscore.name()
                || ast::Ident::with_empty_ctxt(sym).is_path_segment_keyword())
        {
            panic!("`{:?}` is not a valid raw identifier", string)
        }
        Ident { sym, span, is_raw }
    }
}

impl Delimiter {
    pub(crate) fn from_internal(delim: token::DelimToken) -> Delimiter {
        match delim {
            token::Paren => Delimiter::Parenthesis,
            token::Brace => Delimiter::Brace,
            token::Bracket => Delimiter::Bracket,
            token::NoDelim => Delimiter::None,
        }
    }

    pub(crate) fn to_internal(self) -> token::DelimToken {
        match self {
            Delimiter::Parenthesis => token::Paren,
            Delimiter::Brace => token::Brace,
            Delimiter::Bracket => token::Bracket,
            Delimiter::None => token::NoDelim,
        }
    }
}

impl TokenTree {
    pub(crate) fn from_internal(
        stream: tokenstream::TokenStream,
        stack: &mut Vec<TokenTree>,
    ) -> TokenTree {
        use syntax::parse::token::*;

        let (tree, is_joint) = stream.as_tree();
        let (span, token) = match tree {
            tokenstream::TokenTree::Token(span, token) => (span, token),
            tokenstream::TokenTree::Delimited(span, delimed) => {
                let delimiter = Delimiter::from_internal(delimed.delim);
                let mut g = Group::new(delimiter, ::TokenStream(delimed.tts.into()));
                g.set_span(Span(span));
                return g.into();
            }
        };

        let op_kind = if is_joint {
            Spacing::Joint
        } else {
            Spacing::Alone
        };
        macro_rules! tt {
            ($e:expr) => {{
                let mut x = TokenTree::from($e);
                x.set_span(Span(span));
                x
            }};
        }
        macro_rules! op {
            ($a:expr) => {
                tt!(Punct::new($a, op_kind))
            };
            ($a:expr, $b:expr) => {{
                stack.push(tt!(Punct::new($b, op_kind)));
                tt!(Punct::new($a, Spacing::Joint))
            }};
            ($a:expr, $b:expr, $c:expr) => {{
                stack.push(tt!(Punct::new($c, op_kind)));
                stack.push(tt!(Punct::new($b, Spacing::Joint)));
                tt!(Punct::new($a, Spacing::Joint))
            }};
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
            SingleQuote => op!('\''),

            Ident(ident, false) => tt!(self::Ident::new(&ident.as_str(), Span(span))),
            Ident(ident, true) => tt!(self::Ident::new_raw(&ident.as_str(), Span(span))),
            Lifetime(ident) => {
                let ident = ident.without_first_quote();
                stack.push(tt!(self::Ident::new(&ident.as_str(), Span(span))));
                tt!(Punct::new('\'', Spacing::Joint))
            }
            Literal(lit, suffix) => tt!(self::Literal {
                lit,
                suffix,
                span: Span(span)
            }),
            DocComment(c) => {
                let style = comments::doc_comment_style(&c.as_str());
                let stripped = comments::strip_doc_comment_decoration(&c.as_str());
                let stream = vec![
                    tt!(self::Ident::new("doc", Span(span))),
                    tt!(Punct::new('=', Spacing::Alone)),
                    tt!(self::Literal::string(&stripped)),
                ].into_iter()
                    .collect();
                stack.push(tt!(Group::new(Delimiter::Bracket, stream)));
                if style == ast::AttrStyle::Inner {
                    stack.push(tt!(Punct::new('!', Spacing::Alone)));
                }
                tt!(Punct::new('#', Spacing::Alone))
            }

            Interpolated(_) => __internal::with_sess(|sess, _| {
                let tts = token.interpolated_to_tokenstream(sess, span);
                tt!(Group::new(Delimiter::None, ::TokenStream(tts)))
            }),

            DotEq => op!('.', '='),
            OpenDelim(..) | CloseDelim(..) => unreachable!(),
            Whitespace | Comment | Shebang(..) | Eof => unreachable!(),
        }
    }

    pub(crate) fn to_internal(self) -> tokenstream::TokenStream {
        use syntax::parse::token::*;
        use syntax::tokenstream::{Delimited, TokenTree};

        let (ch, kind, span) = match self {
            self::TokenTree::Punct(tt) => (tt.as_char(), tt.spacing(), tt.span()),
            self::TokenTree::Group(tt) => {
                return TokenTree::Delimited(
                    tt.span.0,
                    Delimited {
                        delim: tt.delimiter.to_internal(),
                        tts: tt.stream.0.into(),
                    },
                ).into();
            }
            self::TokenTree::Ident(tt) => {
                let token = Ident(ast::Ident::new(tt.sym, tt.span.0), tt.is_raw);
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
                return vec![a, b].into_iter().collect();
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
                return vec![a, b].into_iter().collect();
            }
            self::TokenTree::Literal(tt) => {
                let token = Literal(tt.lit, tt.suffix);
                return TokenTree::Token(tt.span.0, token).into();
            }
        };

        let token = match ch {
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
            '\'' => SingleQuote,
            _ => unreachable!(),
        };

        let tree = TokenTree::Token(span.0, token);
        match kind {
            Spacing::Alone => tree.into(),
            Spacing::Joint => tree.joint(),
        }
    }
}

impl Level {
    pub(crate) fn to_internal(self) -> errors::Level {
        match self {
            Level::Error => errors::Level::Error,
            Level::Warning => errors::Level::Warning,
            Level::Note => errors::Level::Note,
            Level::Help => errors::Level::Help,
            Level::__Nonexhaustive => unreachable!("Level::__Nonexhaustive"),
        }
    }
}
