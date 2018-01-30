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
use parse::{token, ParseSess};
use print::pprust;
use symbol::keywords;
use syntax_pos::{BytePos, Span, DUMMY_SP};
use tokenstream;

use std::rc::Rc;

/// Contains the sub-token-trees of a "delimited" token tree, such as the contents of `(`. Note
/// that the delimiter itself might be `NoDelim`.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Delimited {
    pub delim: token::DelimToken,
    pub tts: Vec<TokenTree>,
}

impl Delimited {
    /// Return the opening delimiter (possibly `NoDelim`).
    pub fn open_token(&self) -> token::Token {
        token::OpenDelim(self.delim)
    }

    /// Return the closing delimiter (possibly `NoDelim`).
    pub fn close_token(&self) -> token::Token {
        token::CloseDelim(self.delim)
    }

    /// Return a `self::TokenTree` with a `Span` corresponding to the opening delimiter.
    pub fn open_tt(&self, span: Span) -> TokenTree {
        let open_span = if span == DUMMY_SP {
            DUMMY_SP
        } else {
            span.with_lo(span.lo() + BytePos(self.delim.len() as u32))
        };
        TokenTree::Token(open_span, self.open_token())
    }

    /// Return a `self::TokenTree` with a `Span` corresponding to the closing delimiter.
    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = if span == DUMMY_SP {
            DUMMY_SP
        } else {
            span.with_lo(span.hi() - BytePos(self.delim.len() as u32))
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
    /// Kleene star (`*`) for zero or more repetitions
    ZeroOrMore,
    /// Kleene plus (`+`) for one or more repetitions
    OneOrMore,
}

/// Similar to `tokenstream::TokenTree`, except that `$i`, `$i:ident`, and `$(...)`
/// are "first-class" token trees. Useful for parsing macros.
#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum TokenTree {
    Token(Span, token::Token),
    Delimited(Span, Rc<Delimited>),
    /// A kleene-style repetition sequence
    Sequence(Span, Rc<SequenceRepetition>),
    /// E.g. `$var`
    MetaVar(Span, ast::Ident),
    /// E.g. `$var:expr`. This is only used in the left hand side of MBE macros.
    MetaVarDecl(
        Span,
        ast::Ident, /* name to bind */
        ast::Ident, /* kind of nonterminal */
    ),
}

impl TokenTree {
    /// Return the number of tokens in the tree.
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

    /// Returns true if the given token tree contains no other tokens. This is vacuously true for
    /// single tokens or metavar/decls, but may be false for delimited trees or sequences.
    pub fn is_empty(&self) -> bool {
        match *self {
            TokenTree::Delimited(_, ref delimed) => match delimed.delim {
                token::NoDelim => delimed.tts.is_empty(),
                _ => false,
            },
            TokenTree::Sequence(_, ref seq) => seq.tts.is_empty(),
            _ => true,
        }
    }

    /// Get the `index`-th sub-token-tree. This only makes sense for delimited trees and sequences.
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

    /// Retrieve the `TokenTree`'s span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(sp, _)
            | TokenTree::MetaVar(sp, _)
            | TokenTree::MetaVarDecl(sp, _, _)
            | TokenTree::Delimited(sp, _)
            | TokenTree::Sequence(sp, _) => sp,
        }
    }
}

/// Takes a `tokenstream::TokenStream` and returns a `Vec<self::TokenTree>`. Specifically, this
/// takes a generic `TokenStream`, such as is used in the rest of the compiler, and returns a
/// collection of `TokenTree` for use in parsing a macro.
///
/// # Parameters
///
/// - `input`: a token stream to read from, the contents of which we are parsing.
/// - `expect_matchers`: `parse` can be used to parse either the "patterns" or the "body" of a
///   macro. Both take roughly the same form _except_ that in a pattern, metavars are declared with
///   their "matcher" type. For example `$var:expr` or `$id:ident`. In this example, `expr` and
///   `ident` are "matchers". They are not present in the body of a macro rule -- just in the
///   pattern, so we pass a parameter to indicate whether to expect them or not.
/// - `sess`: the parsing session. Any errors will be emitted to this session.
///
/// # Returns
///
/// A collection of `self::TokenTree`. There may also be some errors emitted to `sess`.
pub fn parse(
    input: tokenstream::TokenStream,
    expect_matchers: bool,
    sess: &ParseSess,
) -> Vec<TokenTree> {
    // Will contain the final collection of `self::TokenTree`
    let mut result = Vec::new();

    // For each token tree in `input`, parse the token into a `self::TokenTree`, consuming
    // additional trees if need be.
    let mut trees = input.trees();
    while let Some(tree) = trees.next() {
        let tree = parse_tree(tree, &mut trees, expect_matchers, sess);

        // Given the parsed tree, if there is a metavar and we are expecting matchers, actually
        // parse out the matcher (i.e. in `$id:ident` this would parse the `:` and `ident`).
        match tree {
            TokenTree::MetaVar(start_sp, ident) if expect_matchers => {
                let span = match trees.next() {
                    Some(tokenstream::TokenTree::Token(span, token::Colon)) => match trees.next() {
                        Some(tokenstream::TokenTree::Token(end_sp, ref tok)) => match tok.ident() {
                            Some(kind) => {
                                let span = end_sp.with_lo(start_sp.lo());
                                result.push(TokenTree::MetaVarDecl(span, ident, kind));
                                continue;
                            }
                            _ => end_sp,
                        },
                        tree => tree.as_ref()
                            .map(tokenstream::TokenTree::span)
                            .unwrap_or(span),
                    },
                    tree => tree.as_ref()
                        .map(tokenstream::TokenTree::span)
                        .unwrap_or(start_sp),
                };
                sess.missing_fragment_specifiers.borrow_mut().insert(span);
                result.push(TokenTree::MetaVarDecl(
                    span,
                    ident,
                    keywords::Invalid.ident(),
                ));
            }

            // Not a metavar or no matchers allowed, so just return the tree
            _ => result.push(tree),
        }
    }
    result
}

/// Takes a `tokenstream::TokenTree` and returns a `self::TokenTree`. Specifically, this takes a
/// generic `TokenTree`, such as is used in the rest of the compiler, and returns a `TokenTree`
/// for use in parsing a macro.
///
/// Converting the given tree may involve reading more tokens.
///
/// # Parameters
///
/// - `tree`: the tree we wish to convert.
/// - `trees`: an iterator over trees. We may need to read more tokens from it in order to finish
///   converting `tree`
/// - `expect_matchers`: same as for `parse` (see above).
/// - `sess`: the parsing session. Any errors will be emitted to this session.
fn parse_tree<I>(
    tree: tokenstream::TokenTree,
    trees: &mut I,
    expect_matchers: bool,
    sess: &ParseSess,
) -> TokenTree
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    // Depending on what `tree` is, we could be parsing different parts of a macro
    match tree {
        // `tree` is a `$` token. Look at the next token in `trees`
        tokenstream::TokenTree::Token(span, token::Dollar) => match trees.next() {
            // `tree` is followed by a delimited set of token trees. This indicates the beginning
            // of a repetition sequence in the macro (e.g. `$(pat)*`).
            Some(tokenstream::TokenTree::Delimited(span, delimited)) => {
                // Must have `(` not `{` or `[`
                if delimited.delim != token::Paren {
                    let tok = pprust::token_to_string(&token::OpenDelim(delimited.delim));
                    let msg = format!("expected `(`, found `{}`", tok);
                    sess.span_diagnostic.span_err(span, &msg);
                }
                // Parse the contents of the sequence itself
                let sequence = parse(delimited.tts.into(), expect_matchers, sess);
                // Get the Kleene operator and optional separator
                let (separator, op) = parse_sep_and_kleene_op(trees, span, sess);
                // Count the number of captured "names" (i.e. named metavars)
                let name_captures = macro_parser::count_names(&sequence);
                TokenTree::Sequence(
                    span,
                    Rc::new(SequenceRepetition {
                        tts: sequence,
                        separator,
                        op,
                        num_captures: name_captures,
                    }),
                )
            }

            // `tree` is followed by an `ident`. This could be `$meta_var` or the `$crate` special
            // metavariable that names the crate of the invokation.
            Some(tokenstream::TokenTree::Token(ident_span, ref token)) if token.is_ident() => {
                let ident = token.ident().unwrap();
                let span = ident_span.with_lo(span.lo());
                if ident.name == keywords::Crate.name() {
                    let ident = ast::Ident {
                        name: keywords::DollarCrate.name(),
                        ..ident
                    };
                    TokenTree::Token(span, token::Ident(ident))
                } else {
                    TokenTree::MetaVar(span, ident)
                }
            }

            // `tree` is followed by a random token. This is an error.
            Some(tokenstream::TokenTree::Token(span, tok)) => {
                let msg = format!(
                    "expected identifier, found `{}`",
                    pprust::token_to_string(&tok)
                );
                sess.span_diagnostic.span_err(span, &msg);
                TokenTree::MetaVar(span, keywords::Invalid.ident())
            }

            // There are no more tokens. Just return the `$` we already have.
            None => TokenTree::Token(span, token::Dollar),
        },

        // `tree` is an arbitrary token. Keep it.
        tokenstream::TokenTree::Token(span, tok) => TokenTree::Token(span, tok),

        // `tree` is the beginning of a delimited set of tokens (e.g. `(` or `{`). We need to
        // descend into the delimited set and further parse it.
        tokenstream::TokenTree::Delimited(span, delimited) => TokenTree::Delimited(
            span,
            Rc::new(Delimited {
                delim: delimited.delim,
                tts: parse(delimited.tts.into(), expect_matchers, sess),
            }),
        ),
    }
}

/// Attempt to parse a single Kleene star, possibly with a separator.
///
/// For example, in a pattern such as `$(a),*`, `a` is the pattern to be repeated, `,` is the
/// separator, and `*` is the Kleene operator. This function is specifically concerned with parsing
/// the last two tokens of such a pattern: namely, the optional separator and the Kleene operator
/// itself. Note that here we are parsing the _macro_ itself, rather than trying to match some
/// stream of tokens in an invocation of a macro.
///
/// This function will take some input iterator `input` corresponding to `span` and a parsing
/// session `sess`. If the next one (or possibly two) tokens in `input` correspond to a Kleene
/// operator and separator, then a tuple with `(separator, KleeneOp)` is returned. Otherwise, an
/// error with the appropriate span is emitted to `sess` and a dummy value is returned.
fn parse_sep_and_kleene_op<I>(
    input: &mut I,
    span: Span,
    sess: &ParseSess,
) -> (Option<token::Token>, KleeneOp)
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    fn kleene_op(token: &token::Token) -> Option<KleeneOp> {
        match *token {
            token::BinOp(token::Star) => Some(KleeneOp::ZeroOrMore),
            token::BinOp(token::Plus) => Some(KleeneOp::OneOrMore),
            _ => None,
        }
    }

    // We attempt to look at the next two token trees in `input`. I will call the first #1 and the
    // second #2. If #1 and #2 don't match a valid KleeneOp with/without separator, that is an
    // error, and we should emit an error on the most specific span possible.
    let span = match input.next() {
        // #1 is a token
        Some(tokenstream::TokenTree::Token(span, tok)) => match kleene_op(&tok) {
            // #1 is a KleeneOp with no separator
            Some(op) => return (None, op),

            // #1 is not a KleeneOp, but may be a separator... need to look at #2
            None => match input.next() {
                // #2 is a token
                Some(tokenstream::TokenTree::Token(span, tok2)) => match kleene_op(&tok2) {
                    // #2 is a KleeneOp, so #1 must be a separator
                    Some(op) => return (Some(tok), op),

                    // #2 is not a KleeneOp... error
                    None => span,
                },

                // #2 is not a token at all... error
                tree => tree.as_ref()
                    .map(tokenstream::TokenTree::span)
                    .unwrap_or(span),
            },
        },

        // #1 is not a token at all... error
        tree => tree.as_ref()
            .map(tokenstream::TokenTree::span)
            .unwrap_or(span),
    };

    // Error...
    sess.span_diagnostic.span_err(span, "expected `*` or `+`");
    (None, KleeneOp::ZeroOrMore)
}
