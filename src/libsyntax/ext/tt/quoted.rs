use crate::ast::NodeId;
use crate::early_buffered_lints::BufferedEarlyLintId;
use crate::ext::tt::macro_parser;
use crate::feature_gate::Features;
use crate::parse::token::{self, Token, TokenKind};
use crate::parse::ParseSess;
use crate::print::pprust;
use crate::tokenstream::{self, DelimSpan};
use crate::ast;
use crate::symbol::kw;

use syntax_pos::{edition::Edition, BytePos, Span};

use rustc_data_structures::sync::Lrc;
use std::iter::Peekable;

/// Contains the sub-token-trees of a "delimited" token tree, such as the contents of `(`. Note
/// that the delimiter itself might be `NoDelim`.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub struct Delimited {
    pub delim: token::DelimToken,
    pub tts: Vec<TokenTree>,
}

impl Delimited {
    /// Returns a `self::TokenTree` with a `Span` corresponding to the opening delimiter.
    pub fn open_tt(&self, span: Span) -> TokenTree {
        let open_span = if span.is_dummy() {
            span
        } else {
            span.with_lo(span.lo() + BytePos(self.delim.len() as u32))
        };
        TokenTree::token(token::OpenDelim(self.delim), open_span)
    }

    /// Returns a `self::TokenTree` with a `Span` corresponding to the closing delimiter.
    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = if span.is_dummy() {
            span
        } else {
            span.with_lo(span.hi() - BytePos(self.delim.len() as u32))
        };
        TokenTree::token(token::CloseDelim(self.delim), close_span)
    }
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub struct SequenceRepetition {
    /// The sequence of token trees
    pub tts: Vec<TokenTree>,
    /// The optional separator
    pub separator: Option<Token>,
    /// Whether the sequence can be repeated zero (*), or one or more times (+)
    pub op: KleeneOp,
    /// The number of `Match`s that appear in the sequence (and subsequences)
    pub num_captures: usize,
}

/// A Kleene-style [repetition operator](http://en.wikipedia.org/wiki/Kleene_star)
/// for token sequences.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum KleeneOp {
    /// Kleene star (`*`) for zero or more repetitions
    ZeroOrMore,
    /// Kleene plus (`+`) for one or more repetitions
    OneOrMore,
    /// Kleene optional (`?`) for zero or one reptitions
    ZeroOrOne,
}

/// Similar to `tokenstream::TokenTree`, except that `$i`, `$i:ident`, and `$(...)`
/// are "first-class" token trees. Useful for parsing macros.
#[derive(Debug, Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum TokenTree {
    Token(Token),
    Delimited(DelimSpan, Lrc<Delimited>),
    /// A kleene-style repetition sequence
    Sequence(DelimSpan, Lrc<SequenceRepetition>),
    /// e.g., `$var`
    MetaVar(Span, ast::Ident),
    /// e.g., `$var:expr`. This is only used in the left hand side of MBE macros.
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

    /// Returns `true` if the given token tree contains no other tokens. This is vacuously true for
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

    /// Gets the `index`-th sub-token-tree. This only makes sense for delimited trees and sequences.
    pub fn get_tt(&self, index: usize) -> TokenTree {
        match (self, index) {
            (&TokenTree::Delimited(_, ref delimed), _) if delimed.delim == token::NoDelim => {
                delimed.tts[index].clone()
            }
            (&TokenTree::Delimited(span, ref delimed), _) => {
                if index == 0 {
                    return delimed.open_tt(span.open);
                }
                if index == delimed.tts.len() + 1 {
                    return delimed.close_tt(span.close);
                }
                delimed.tts[index - 1].clone()
            }
            (&TokenTree::Sequence(_, ref seq), _) => seq.tts[index].clone(),
            _ => panic!("Cannot expand a token tree"),
        }
    }

    /// Retrieves the `TokenTree`'s span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(Token { span, .. })
            | TokenTree::MetaVar(span, _)
            | TokenTree::MetaVarDecl(span, _, _) => span,
            TokenTree::Delimited(span, _)
            | TokenTree::Sequence(span, _) => span.entire(),
        }
    }

    crate fn token(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span))
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
/// - `features`, `attrs`: language feature flags and attributes so that we know whether to use
///   unstable features or not.
/// - `edition`: which edition are we in.
/// - `macro_node_id`: the NodeId of the macro we are parsing.
///
/// # Returns
///
/// A collection of `self::TokenTree`. There may also be some errors emitted to `sess`.
pub fn parse(
    input: tokenstream::TokenStream,
    expect_matchers: bool,
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    edition: Edition,
    macro_node_id: NodeId,
) -> Vec<TokenTree> {
    // Will contain the final collection of `self::TokenTree`
    let mut result = Vec::new();

    // For each token tree in `input`, parse the token into a `self::TokenTree`, consuming
    // additional trees if need be.
    let mut trees = input.trees().peekable();
    while let Some(tree) = trees.next() {
        // Given the parsed tree, if there is a metavar and we are expecting matchers, actually
        // parse out the matcher (i.e., in `$id:ident` this would parse the `:` and `ident`).
        let tree = parse_tree(
            tree,
            &mut trees,
            expect_matchers,
            sess,
            features,
            attrs,
            edition,
            macro_node_id,
        );
        match tree {
            TokenTree::MetaVar(start_sp, ident) if expect_matchers => {
                let span = match trees.next() {
                    Some(tokenstream::TokenTree::Token(Token { kind: token::Colon, span })) =>
                        match trees.next() {
                            Some(tokenstream::TokenTree::Token(token)) => match token.ident() {
                                Some((kind, _)) => {
                                    let span = token.span.with_lo(start_sp.lo());
                                    result.push(TokenTree::MetaVarDecl(span, ident, kind));
                                    continue;
                                }
                                _ => token.span,
                            },
                            tree => tree
                                .as_ref()
                                .map(tokenstream::TokenTree::span)
                                .unwrap_or(span),
                        },
                    tree => tree
                        .as_ref()
                        .map(tokenstream::TokenTree::span)
                        .unwrap_or(start_sp),
                };
                sess.missing_fragment_specifiers.borrow_mut().insert(span);
                result.push(TokenTree::MetaVarDecl(
                    span,
                    ident,
                    ast::Ident::invalid(),
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
/// - `features`, `attrs`: language feature flags and attributes so that we know whether to use
///   unstable features or not.
fn parse_tree<I>(
    tree: tokenstream::TokenTree,
    trees: &mut Peekable<I>,
    expect_matchers: bool,
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    edition: Edition,
    macro_node_id: NodeId,
) -> TokenTree
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    // Depending on what `tree` is, we could be parsing different parts of a macro
    match tree {
        // `tree` is a `$` token. Look at the next token in `trees`
        tokenstream::TokenTree::Token(Token { kind: token::Dollar, span }) => match trees.next() {
            // `tree` is followed by a delimited set of token trees. This indicates the beginning
            // of a repetition sequence in the macro (e.g. `$(pat)*`).
            Some(tokenstream::TokenTree::Delimited(span, delim, tts)) => {
                // Must have `(` not `{` or `[`
                if delim != token::Paren {
                    let tok = pprust::token_kind_to_string(&token::OpenDelim(delim));
                    let msg = format!("expected `(`, found `{}`", tok);
                    sess.span_diagnostic.span_err(span.entire(), &msg);
                }
                // Parse the contents of the sequence itself
                let sequence = parse(
                    tts.into(),
                    expect_matchers,
                    sess,
                    features,
                    attrs,
                    edition,
                    macro_node_id,
                );
                // Get the Kleene operator and optional separator
                let (separator, op) =
                    parse_sep_and_kleene_op(
                        trees,
                        span.entire(),
                        sess,
                        features,
                        attrs,
                        edition,
                        macro_node_id,
                    );
                // Count the number of captured "names" (i.e., named metavars)
                let name_captures = macro_parser::count_names(&sequence);
                TokenTree::Sequence(
                    span,
                    Lrc::new(SequenceRepetition {
                        tts: sequence,
                        separator,
                        op,
                        num_captures: name_captures,
                    }),
                )
            }

            // `tree` is followed by an `ident`. This could be `$meta_var` or the `$crate` special
            // metavariable that names the crate of the invocation.
            Some(tokenstream::TokenTree::Token(token)) if token.is_ident() => {
                let (ident, is_raw) = token.ident().unwrap();
                let span = ident.span.with_lo(span.lo());
                if ident.name == kw::Crate && !is_raw {
                    TokenTree::token(token::Ident(kw::DollarCrate, is_raw), span)
                } else {
                    TokenTree::MetaVar(span, ident)
                }
            }

            // `tree` is followed by a random token. This is an error.
            Some(tokenstream::TokenTree::Token(token)) => {
                let msg = format!(
                    "expected identifier, found `{}`",
                    pprust::token_to_string(&token),
                );
                sess.span_diagnostic.span_err(token.span, &msg);
                TokenTree::MetaVar(token.span, ast::Ident::invalid())
            }

            // There are no more tokens. Just return the `$` we already have.
            None => TokenTree::token(token::Dollar, span),
        },

        // `tree` is an arbitrary token. Keep it.
        tokenstream::TokenTree::Token(token) => TokenTree::Token(token),

        // `tree` is the beginning of a delimited set of tokens (e.g., `(` or `{`). We need to
        // descend into the delimited set and further parse it.
        tokenstream::TokenTree::Delimited(span, delim, tts) => TokenTree::Delimited(
            span,
            Lrc::new(Delimited {
                delim: delim,
                tts: parse(
                    tts.into(),
                    expect_matchers,
                    sess,
                    features,
                    attrs,
                    edition,
                    macro_node_id,
                ),
            }),
        ),
    }
}

/// Takes a token and returns `Some(KleeneOp)` if the token is `+` `*` or `?`. Otherwise, return
/// `None`.
fn kleene_op(token: &Token) -> Option<KleeneOp> {
    match token.kind {
        token::BinOp(token::Star) => Some(KleeneOp::ZeroOrMore),
        token::BinOp(token::Plus) => Some(KleeneOp::OneOrMore),
        token::Question => Some(KleeneOp::ZeroOrOne),
        _ => None,
    }
}

/// Parse the next token tree of the input looking for a KleeneOp. Returns
///
/// - Ok(Ok((op, span))) if the next token tree is a KleeneOp
/// - Ok(Err(tok, span)) if the next token tree is a token but not a KleeneOp
/// - Err(span) if the next token tree is not a token
fn parse_kleene_op<I>(input: &mut I, span: Span) -> Result<Result<(KleeneOp, Span), Token>, Span>
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    match input.next() {
        Some(tokenstream::TokenTree::Token(token)) => match kleene_op(&token) {
            Some(op) => Ok(Ok((op, token.span))),
            None => Ok(Err(token)),
        },
        tree => Err(tree
            .as_ref()
            .map(tokenstream::TokenTree::span)
            .unwrap_or(span)),
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
///
/// N.B., in the 2015 edition, `*` and `+` are the only Kleene operators, and `?` is a separator.
/// In the 2018 edition however, `?` is a Kleene operator, and not a separator.
fn parse_sep_and_kleene_op<I>(
    input: &mut Peekable<I>,
    span: Span,
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    edition: Edition,
    macro_node_id: NodeId,
) -> (Option<Token>, KleeneOp)
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    match edition {
        Edition::Edition2015 => parse_sep_and_kleene_op_2015(
            input,
            span,
            sess,
            features,
            attrs,
            macro_node_id,
        ),
        Edition::Edition2018 => parse_sep_and_kleene_op_2018(input, span, sess, features, attrs),
    }
}

// `?` is a separator (with a migration warning) and never a KleeneOp.
fn parse_sep_and_kleene_op_2015<I>(
    input: &mut Peekable<I>,
    span: Span,
    sess: &ParseSess,
    _features: &Features,
    _attrs: &[ast::Attribute],
    macro_node_id: NodeId,
) -> (Option<Token>, KleeneOp)
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    // We basically look at two token trees here, denoted as #1 and #2 below
    let span = match parse_kleene_op(input, span) {
        // #1 is a `+` or `*` KleeneOp
        //
        // `?` is ambiguous: it could be a separator (warning) or a Kleene::ZeroOrOne (error), so
        // we need to look ahead one more token to be sure.
        Ok(Ok((op, _))) if op != KleeneOp::ZeroOrOne => return (None, op),

        // #1 is `?` token, but it could be a Kleene::ZeroOrOne (error in 2015) without a separator
        // or it could be a `?` separator followed by any Kleene operator. We need to look ahead 1
        // token to find out which.
        Ok(Ok((op, op1_span))) => {
            assert_eq!(op, KleeneOp::ZeroOrOne);

            // Lookahead at #2. If it is a KleenOp, then #1 is a separator.
            let is_1_sep = if let Some(tokenstream::TokenTree::Token(tok2)) = input.peek() {
                kleene_op(tok2).is_some()
            } else {
                false
            };

            if is_1_sep {
                // #1 is a separator and #2 should be a KleepeOp.
                // (N.B. We need to advance the input iterator.)
                match parse_kleene_op(input, span) {
                    // #2 is `?`, which is not allowed as a Kleene op in 2015 edition,
                    // but is allowed in the 2018 edition.
                    Ok(Ok((op, op2_span))) if op == KleeneOp::ZeroOrOne => {
                        sess.span_diagnostic
                            .struct_span_err(op2_span, "expected `*` or `+`")
                            .note("`?` is not a macro repetition operator in the 2015 edition, \
                                 but is accepted in the 2018 edition")
                            .emit();

                        // Return a dummy
                        return (None, KleeneOp::ZeroOrMore);
                    }

                    // #2 is a Kleene op, which is the only valid option
                    Ok(Ok((op, _))) => {
                        // Warn that `?` as a separator will be deprecated
                        sess.buffer_lint(
                            BufferedEarlyLintId::QuestionMarkMacroSep,
                            op1_span,
                            macro_node_id,
                            "using `?` as a separator is deprecated and will be \
                             a hard error in an upcoming edition",
                        );

                        return (Some(Token::new(token::Question, op1_span)), op);
                    }

                    // #2 is a random token (this is an error) :(
                    Ok(Err(_)) => op1_span,

                    // #2 is not even a token at all :(
                    Err(_) => op1_span,
                }
            } else {
                // `?` is not allowed as a Kleene op in 2015,
                // but is allowed in the 2018 edition
                sess.span_diagnostic
                    .struct_span_err(op1_span, "expected `*` or `+`")
                    .note("`?` is not a macro repetition operator in the 2015 edition, \
                         but is accepted in the 2018 edition")
                    .emit();

                // Return a dummy
                return (None, KleeneOp::ZeroOrMore);
            }
        }

        // #1 is a separator followed by #2, a KleeneOp
        Ok(Err(token)) => match parse_kleene_op(input, token.span) {
            // #2 is a `?`, which is not allowed as a Kleene op in 2015 edition,
            // but is allowed in the 2018 edition
            Ok(Ok((op, op2_span))) if op == KleeneOp::ZeroOrOne => {
                sess.span_diagnostic
                    .struct_span_err(op2_span, "expected `*` or `+`")
                    .note("`?` is not a macro repetition operator in the 2015 edition, \
                        but is accepted in the 2018 edition")
                    .emit();

                // Return a dummy
                return (None, KleeneOp::ZeroOrMore);
            }

            // #2 is a KleeneOp :D
            Ok(Ok((op, _))) => return (Some(token), op),

            // #2 is a random token :(
            Ok(Err(token)) => token.span,

            // #2 is not a token at all :(
            Err(span) => span,
        },

        // #1 is not a token
        Err(span) => span,
    };

    sess.span_diagnostic.span_err(span, "expected `*` or `+`");

    // Return a dummy
    (None, KleeneOp::ZeroOrMore)
}

// `?` is a Kleene op, not a separator
fn parse_sep_and_kleene_op_2018<I>(
    input: &mut Peekable<I>,
    span: Span,
    sess: &ParseSess,
    _features: &Features,
    _attrs: &[ast::Attribute],
) -> (Option<Token>, KleeneOp)
where
    I: Iterator<Item = tokenstream::TokenTree>,
{
    // We basically look at two token trees here, denoted as #1 and #2 below
    let span = match parse_kleene_op(input, span) {
        // #1 is a `?` (needs feature gate)
        Ok(Ok((op, _op1_span))) if op == KleeneOp::ZeroOrOne => {
            return (None, op);
        }

        // #1 is a `+` or `*` KleeneOp
        Ok(Ok((op, _))) => return (None, op),

        // #1 is a separator followed by #2, a KleeneOp
        Ok(Err(token)) => match parse_kleene_op(input, token.span) {
            // #2 is the `?` Kleene op, which does not take a separator (error)
            Ok(Ok((op, _op2_span))) if op == KleeneOp::ZeroOrOne => {
                // Error!
                sess.span_diagnostic.span_err(
                    token.span,
                    "the `?` macro repetition operator does not take a separator",
                );

                // Return a dummy
                return (None, KleeneOp::ZeroOrMore);
            }

            // #2 is a KleeneOp :D
            Ok(Ok((op, _))) => return (Some(token), op),

            // #2 is a random token :(
            Ok(Err(token)) => token.span,

            // #2 is not a token at all :(
            Err(span) => span,
        },

        // #1 is not a token
        Err(span) => span,
    };

    // If we ever get to this point, we have experienced an "unexpected token" error
    sess.span_diagnostic
        .span_err(span, "expected one of: `*`, `+`, or `?`");

    // Return a dummy
    (None, KleeneOp::ZeroOrMore)
}
