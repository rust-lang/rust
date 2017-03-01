// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ext::tt::macro_parser;
use parse::{ParseSess, token};
use print::pprust;
use symbol::{keywords, Symbol};
use syntax_pos::{DUMMY_SP, Span, BytePos};
use tokenstream;

use std::rc::Rc;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Delimited {
    pub delim: token::DelimToken,
    pub tts: Vec<TokenTree>,
}

impl Delimited {
    pub fn open_token(&self) -> token::Token {
        token::OpenDelim(self.delim)
    }

    pub fn close_token(&self) -> token::Token {
        token::CloseDelim(self.delim)
    }

    pub fn open_tt(&self, span: Span) -> TokenTree {
        let open_span = match span {
            DUMMY_SP => DUMMY_SP,
            _ => Span { hi: span.lo + BytePos(self.delim.len() as u32), ..span },
        };
        TokenTree::Token(open_span, self.open_token())
    }

    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = match span {
            DUMMY_SP => DUMMY_SP,
            _ => Span { lo: span.hi - BytePos(self.delim.len() as u32), ..span },
        };
        TokenTree::Token(close_span, self.close_token())
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct SequenceRepetition {
    /// The sequence of token trees
    pub tts: Vec<TokenTree>,
    /// The optional separator
    pub separator: Option<token::Token>,
    /// Whether the sequence can be repeated zero (*), or one or more times (+)
    pub op: KleeneOp,
    /// The number of `Match`s that appear in the sequence (and subsequences)
    pub num_captures: usize,
}

/// A Kleene-style [repetition operator](http://en.wikipedia.org/wiki/Kleene_star)
/// for token sequences.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum KleeneOp {
    ZeroOrMore,
    OneOrMore,
}

/// Similar to `tokenstream::TokenTree`, except that `$i`, `$i:ident`, and `$(...)`
/// are "first-class" token trees.
#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum TokenTree {
    Token(Span, token::Token),
    Delimited(Span, Rc<Delimited>),
    /// A kleene-style repetition sequence with a span
    Sequence(Span, Rc<SequenceRepetition>),
    /// Matches a nonterminal. This is only used in the left hand side of MBE macros.
    MetaVarDecl(Span, ast::Ident /* name to bind */, ast::Ident /* kind of nonterminal */),
}

impl TokenTree {
    pub fn len(&self) -> usize {
        match *self {
            TokenTree::Delimited(_, ref delimed) => match delimed.delim {
                token::NoDelim => delimed.tts.len(),
                _ => delimed.tts.len() + 2,
            },
            TokenTree::Sequence(_, ref seq) => seq.tts.len(),
            _ => 0,
        }
    }

    pub fn get_tt(&self, index: usize) -> TokenTree {
        match (self, index) {
            (&TokenTree::Delimited(_, ref delimed), _) if delimed.delim == token::NoDelim => {
                delimed.tts[index].clone()
            }
            (&TokenTree::Delimited(span, ref delimed), _) => {
                if index == 0 {
                    return delimed.open_tt(span);
                }
                if index == delimed.tts.len() + 1 {
                    return delimed.close_tt(span);
                }
                delimed.tts[index - 1].clone()
            }
            (&TokenTree::Sequence(_, ref seq), _) => seq.tts[index].clone(),
            _ => panic!("Cannot expand a token tree"),
        }
    }

    /// Retrieve the TokenTree's span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(sp, _) |
            TokenTree::MetaVarDecl(sp, _, _) |
            TokenTree::Delimited(sp, _) |
            TokenTree::Sequence(sp, _) => sp,
        }
    }
}

pub fn parse(input: &[tokenstream::TokenTree], expect_matchers: bool, sess: &ParseSess)
             -> Vec<TokenTree> {
    let mut result = Vec::new();
    let mut trees = input.iter().cloned();
    while let Some(tree) = trees.next() {
        let tree = parse_tree(tree, &mut trees, expect_matchers, sess);
        match tree {
            TokenTree::Token(start_sp, token::SubstNt(ident)) if expect_matchers => {
                let span = match trees.next() {
                    Some(tokenstream::TokenTree::Token(span, token::Colon)) => match trees.next() {
                        Some(tokenstream::TokenTree::Token(end_sp, token::Ident(kind))) => {
                            let span = Span { lo: start_sp.lo, ..end_sp };
                            result.push(TokenTree::MetaVarDecl(span, ident, kind));
                            continue
                        }
                        tree @ _ => tree.as_ref().map(tokenstream::TokenTree::span).unwrap_or(span),
                    },
                    tree @ _ => tree.as_ref().map(tokenstream::TokenTree::span).unwrap_or(start_sp),
                };
                sess.missing_fragment_specifiers.borrow_mut().insert(span);
                result.push(TokenTree::MetaVarDecl(span, ident, keywords::Invalid.ident()));
            }
            _ => result.push(tree),
        }
    }
    result
}

fn parse_tree<I>(tree: tokenstream::TokenTree,
                 trees: &mut I,
                 expect_matchers: bool,
                 sess: &ParseSess)
                 -> TokenTree
    where I: Iterator<Item = tokenstream::TokenTree>,
{
    match tree {
        tokenstream::TokenTree::Token(span, token::Dollar) => match trees.next() {
            Some(tokenstream::TokenTree::Delimited(span, ref delimited)) => {
                if delimited.delim != token::Paren {
                    let tok = pprust::token_to_string(&token::OpenDelim(delimited.delim));
                    let msg = format!("expected `(`, found `{}`", tok);
                    sess.span_diagnostic.span_err(span, &msg);
                }
                let sequence = parse(&delimited.tts, expect_matchers, sess);
                let (separator, op) = parse_sep_and_kleene_op(trees, span, sess);
                let name_captures = macro_parser::count_names(&sequence);
                TokenTree::Sequence(span, Rc::new(SequenceRepetition {
                    tts: sequence,
                    separator: separator,
                    op: op,
                    num_captures: name_captures,
                }))
            }
            Some(tokenstream::TokenTree::Token(ident_span, token::Ident(ident))) => {
                let span = Span { lo: span.lo, ..ident_span };
                if ident.name == keywords::Crate.name() {
                    let ident = ast::Ident { name: Symbol::intern("$crate"), ..ident };
                    TokenTree::Token(span, token::Ident(ident))
                } else {
                    TokenTree::Token(span, token::SubstNt(ident))
                }
            }
            Some(tokenstream::TokenTree::Token(span, tok)) => {
                let msg = format!("expected identifier, found `{}`", pprust::token_to_string(&tok));
                sess.span_diagnostic.span_err(span, &msg);
                TokenTree::Token(span, token::SubstNt(keywords::Invalid.ident()))
            }
            None => TokenTree::Token(span, token::Dollar),
        },
        tokenstream::TokenTree::Token(span, tok) => TokenTree::Token(span, tok),
        tokenstream::TokenTree::Delimited(span, delimited) => {
            TokenTree::Delimited(span, Rc::new(Delimited {
                delim: delimited.delim,
                tts: parse(&delimited.tts, expect_matchers, sess),
            }))
        }
    }
}

fn parse_sep_and_kleene_op<I>(input: &mut I, span: Span, sess: &ParseSess)
                              -> (Option<token::Token>, KleeneOp)
    where I: Iterator<Item = tokenstream::TokenTree>,
{
    fn kleene_op(token: &token::Token) -> Option<KleeneOp> {
        match *token {
            token::BinOp(token::Star) => Some(KleeneOp::ZeroOrMore),
            token::BinOp(token::Plus) => Some(KleeneOp::OneOrMore),
            _ => None,
        }
    }

    let span = match input.next() {
        Some(tokenstream::TokenTree::Token(span, tok)) => match kleene_op(&tok) {
            Some(op) => return (None, op),
            None => match input.next() {
                Some(tokenstream::TokenTree::Token(span, tok2)) => match kleene_op(&tok2) {
                    Some(op) => return (Some(tok), op),
                    None => span,
                },
                tree @ _ => tree.as_ref().map(tokenstream::TokenTree::span).unwrap_or(span),
            }
        },
        tree @ _ => tree.as_ref().map(tokenstream::TokenTree::span).unwrap_or(span),
    };

    sess.span_diagnostic.span_err(span, "expected `*` or `+`");
    (None, KleeneOp::ZeroOrMore)
}
