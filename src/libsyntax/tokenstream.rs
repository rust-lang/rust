// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Token Trees
//! TokenTrees are syntactic forms for dealing with tokens. The description below is
//! more complete; in short a TokenTree is a single token, a delimited sequence of token
//! trees, or a sequence with repetition for list splicing as part of macro expansion.

use ast::{AttrStyle};
use codemap::{Span};
use ext::base;
use ext::tt::macro_parser;
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::lexer;
use parse::token;

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
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TokenTree {
    /// A single token
    Token(Span, token::Token),
    /// A delimited sequence of token trees
    Delimited(Span, Delimited),

    // This only makes sense in MBE macros.

    /// A kleene-style repetition sequence with a span
    // FIXME(eddyb) #12938 Use DST.
    Sequence(Span, SequenceRepetition),
}

impl TokenTree {
    pub fn len(&self) -> usize {
        match *self {
            TokenTree::Token(_, token::DocComment(name)) => {
                match doc_comment_style(&name.as_str()) {
                    AttrStyle::Outer => 2,
                    AttrStyle::Inner => 3
                }
            }
            TokenTree::Token(_, token::SpecialVarNt(..)) => 2,
            TokenTree::Token(_, token::MatchNt(..)) => 3,
            TokenTree::Delimited(_, ref delimed) => {
                delimed.tts.len() + 2
            }
            TokenTree::Sequence(_, ref seq) => {
                seq.tts.len()
            }
            TokenTree::Token(..) => 0
        }
    }

    pub fn get_tt(&self, index: usize) -> TokenTree {
        match (self, index) {
            (&TokenTree::Token(sp, token::DocComment(_)), 0) => {
                TokenTree::Token(sp, token::Pound)
            }
            (&TokenTree::Token(sp, token::DocComment(name)), 1)
            if doc_comment_style(&name.as_str()) == AttrStyle::Inner => {
                TokenTree::Token(sp, token::Not)
            }
            (&TokenTree::Token(sp, token::DocComment(name)), _) => {
                let stripped = strip_doc_comment_decoration(&name.as_str());

                // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
                // required to wrap the text.
                let num_of_hashes = stripped.chars().scan(0, |cnt, x| {
                    *cnt = if x == '"' {
                        1
                    } else if *cnt != 0 && x == '#' {
                        *cnt + 1
                    } else {
                        0
                    };
                    Some(*cnt)
                }).max().unwrap_or(0);

                TokenTree::Delimited(sp, Delimited {
                    delim: token::Bracket,
                    open_span: sp,
                    tts: vec![TokenTree::Token(sp, token::Ident(token::str_to_ident("doc"))),
                              TokenTree::Token(sp, token::Eq),
                              TokenTree::Token(sp, token::Literal(
                                  token::StrRaw(token::intern(&stripped), num_of_hashes), None))],
                    close_span: sp,
                })
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
            (&TokenTree::Sequence(_, ref seq), _) => {
                seq.tts[index].clone()
            }
            _ => panic!("Cannot expand a token tree")
        }
    }

    /// Returns the `Span` corresponding to this token tree.
    pub fn get_span(&self) -> Span {
        match *self {
            TokenTree::Token(span, _)     => span,
            TokenTree::Delimited(span, _) => span,
            TokenTree::Sequence(span, _)  => span,
        }
    }

    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt, mtch: &[TokenTree], tts: &[TokenTree])
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let arg_rdr = lexer::new_tt_reader_with_doc_flag(&cx.parse_sess().span_diagnostic,
                                                         None,
                                                         None,
                                                         tts.iter().cloned().collect(),
                                                         true);
        macro_parser::parse(cx.parse_sess(), cx.cfg(), arg_rdr, mtch)
    }
}

