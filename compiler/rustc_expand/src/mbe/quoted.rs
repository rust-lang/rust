use rustc_ast::token::{self, Delimiter, IdentIsRaw, NonterminalKind, Token};
use rustc_ast::tokenstream::TokenStreamIter;
use rustc_ast::{NodeId, tokenstream};
use rustc_ast_pretty::pprust;
use rustc_feature::Features;
use rustc_session::Session;
use rustc_session::parse::feature_err;
use rustc_span::edition::Edition;
use rustc_span::{Ident, Span, kw, sym};

use crate::errors;
use crate::mbe::macro_parser::count_metavar_decls;
use crate::mbe::{Delimited, KleeneOp, KleeneToken, MetaVarExpr, SequenceRepetition, TokenTree};

pub(crate) const VALID_FRAGMENT_NAMES_MSG: &str = "valid fragment specifiers are \
    `ident`, `block`, `stmt`, `expr`, `pat`, `ty`, `lifetime`, `literal`, `path`, \
    `meta`, `tt`, `item` and `vis`, along with `expr_2021` and `pat_param` for edition compatibility";

/// Takes a `tokenstream::TokenStream` and returns a `Vec<self::TokenTree>`. Specifically, this
/// takes a generic `TokenStream`, such as is used in the rest of the compiler, and returns a
/// collection of `TokenTree` for use in parsing a macro.
///
/// # Parameters
///
/// - `input`: a token stream to read from, the contents of which we are parsing.
/// - `parsing_patterns`: `parse` can be used to parse either the "patterns" or the "body" of a
///   macro. Both take roughly the same form _except_ that:
///   - In a pattern, metavars are declared with their "matcher" type. For example `$var:expr` or
///     `$id:ident`. In this example, `expr` and `ident` are "matchers". They are not present in the
///     body of a macro rule -- just in the pattern.
///   - Metavariable expressions are only valid in the "body", not the "pattern".
/// - `sess`: the parsing session. Any errors will be emitted to this session.
/// - `node_id`: the NodeId of the macro we are parsing.
/// - `features`: language features so we can do feature gating.
///
/// # Returns
///
/// A collection of `self::TokenTree`. There may also be some errors emitted to `sess`.
pub(super) fn parse(
    input: &tokenstream::TokenStream,
    parsing_patterns: bool,
    sess: &Session,
    node_id: NodeId,
    features: &Features,
    edition: Edition,
) -> Vec<TokenTree> {
    // Will contain the final collection of `self::TokenTree`
    let mut result = Vec::new();

    // For each token tree in `input`, parse the token into a `self::TokenTree`, consuming
    // additional trees if need be.
    let mut iter = input.iter();
    while let Some(tree) = iter.next() {
        // Given the parsed tree, if there is a metavar and we are expecting matchers, actually
        // parse out the matcher (i.e., in `$id:ident` this would parse the `:` and `ident`).
        let tree = parse_tree(tree, &mut iter, parsing_patterns, sess, node_id, features, edition);
        match tree {
            TokenTree::MetaVar(start_sp, ident) if parsing_patterns => {
                // Not consuming the next token immediately, as it may not be a colon
                let span = match iter.peek() {
                    Some(&tokenstream::TokenTree::Token(
                        Token { kind: token::Colon, span: colon_span },
                        _,
                    )) => {
                        // Consume the colon first
                        iter.next();

                        // It's ok to consume the next tree no matter how,
                        // since if it's not a token then it will be an invalid declaration.
                        match iter.next() {
                            Some(tokenstream::TokenTree::Token(token, _)) => match token.ident() {
                                Some((fragment, _)) => {
                                    let span = token.span.with_lo(start_sp.lo());
                                    let edition = || {
                                        // FIXME(#85708) - once we properly decode a foreign
                                        // crate's `SyntaxContext::root`, then we can replace
                                        // this with just `span.edition()`. A
                                        // `SyntaxContext::root()` from the current crate will
                                        // have the edition of the current crate, and a
                                        // `SyntaxContext::root()` from a foreign crate will
                                        // have the edition of that crate (which we manually
                                        // retrieve via the `edition` parameter).
                                        if !span.from_expansion() {
                                            edition
                                        } else {
                                            span.edition()
                                        }
                                    };
                                    let kind = NonterminalKind::from_symbol(fragment.name, edition)
                                        .unwrap_or_else(|| {
                                            sess.dcx().emit_err(errors::InvalidFragmentSpecifier {
                                                span,
                                                fragment,
                                                help: VALID_FRAGMENT_NAMES_MSG.into(),
                                            });
                                            NonterminalKind::Ident
                                        });
                                    result.push(TokenTree::MetaVarDecl(span, ident, Some(kind)));
                                    continue;
                                }
                                _ => token.span,
                            },
                            // Invalid, return a nice source location
                            _ => colon_span.with_lo(start_sp.lo()),
                        }
                    }
                    // Whether it's none or some other tree, it doesn't belong to
                    // the current meta variable, returning the original span.
                    _ => start_sp,
                };

                result.push(TokenTree::MetaVarDecl(span, ident, None));
            }

            // Not a metavar or no matchers allowed, so just return the tree
            _ => result.push(tree),
        }
    }
    result
}

/// Asks for the `macro_metavar_expr` feature if it is not enabled
fn maybe_emit_macro_metavar_expr_feature(features: &Features, sess: &Session, span: Span) {
    if !features.macro_metavar_expr() {
        let msg = "meta-variable expressions are unstable";
        feature_err(sess, sym::macro_metavar_expr, span, msg).emit();
    }
}

fn maybe_emit_macro_metavar_expr_concat_feature(features: &Features, sess: &Session, span: Span) {
    if !features.macro_metavar_expr_concat() {
        let msg = "the `concat` meta-variable expression is unstable";
        feature_err(sess, sym::macro_metavar_expr_concat, span, msg).emit();
    }
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
/// - `outer_iter`: an iterator over trees. We may need to read more tokens from it in order to finish
///   converting `tree`
/// - `parsing_patterns`: same as [parse].
/// - `sess`: the parsing session. Any errors will be emitted to this session.
/// - `features`: language features so we can do feature gating.
fn parse_tree<'a>(
    tree: &'a tokenstream::TokenTree,
    outer_iter: &mut TokenStreamIter<'a>,
    parsing_patterns: bool,
    sess: &Session,
    node_id: NodeId,
    features: &Features,
    edition: Edition,
) -> TokenTree {
    // Depending on what `tree` is, we could be parsing different parts of a macro
    match tree {
        // `tree` is a `$` token. Look at the next token in `trees`
        &tokenstream::TokenTree::Token(Token { kind: token::Dollar, span: dollar_span }, _) => {
            // FIXME: Handle `Invisible`-delimited groups in a more systematic way
            // during parsing.
            let mut next = outer_iter.next();
            let mut iter_storage;
            let mut iter: &mut TokenStreamIter<'_> = match next {
                Some(tokenstream::TokenTree::Delimited(.., delim, tts)) if delim.skip() => {
                    iter_storage = tts.iter();
                    next = iter_storage.next();
                    &mut iter_storage
                }
                _ => outer_iter,
            };

            match next {
                // `tree` is followed by a delimited set of token trees.
                Some(&tokenstream::TokenTree::Delimited(delim_span, _, delim, ref tts)) => {
                    if parsing_patterns {
                        if delim != Delimiter::Parenthesis {
                            span_dollar_dollar_or_metavar_in_the_lhs_err(
                                sess,
                                &Token {
                                    kind: delim.as_open_token_kind(),
                                    span: delim_span.entire(),
                                },
                            );
                        }
                    } else {
                        match delim {
                            Delimiter::Brace => {
                                // The delimiter is `{`. This indicates the beginning
                                // of a meta-variable expression (e.g. `${count(ident)}`).
                                // Try to parse the meta-variable expression.
                                match MetaVarExpr::parse(tts, delim_span.entire(), &sess.psess) {
                                    Err(err) => {
                                        err.emit();
                                        // Returns early the same read `$` to avoid spanning
                                        // unrelated diagnostics that could be performed afterwards
                                        return TokenTree::token(token::Dollar, dollar_span);
                                    }
                                    Ok(elem) => {
                                        if let MetaVarExpr::Concat(_) = elem {
                                            maybe_emit_macro_metavar_expr_concat_feature(
                                                features,
                                                sess,
                                                delim_span.entire(),
                                            );
                                        } else {
                                            maybe_emit_macro_metavar_expr_feature(
                                                features,
                                                sess,
                                                delim_span.entire(),
                                            );
                                        }
                                        return TokenTree::MetaVarExpr(delim_span, elem);
                                    }
                                }
                            }
                            Delimiter::Parenthesis => {}
                            _ => {
                                let token =
                                    pprust::token_kind_to_string(&delim.as_open_token_kind());
                                sess.dcx().emit_err(errors::ExpectedParenOrBrace {
                                    span: delim_span.entire(),
                                    token,
                                });
                            }
                        }
                    }
                    // If we didn't find a metavar expression above, then we must have a
                    // repetition sequence in the macro (e.g. `$(pat)*`). Parse the
                    // contents of the sequence itself
                    let sequence = parse(tts, parsing_patterns, sess, node_id, features, edition);
                    // Get the Kleene operator and optional separator
                    let (separator, kleene) =
                        parse_sep_and_kleene_op(&mut iter, delim_span.entire(), sess);
                    // Count the number of captured "names" (i.e., named metavars)
                    let num_captures =
                        if parsing_patterns { count_metavar_decls(&sequence) } else { 0 };
                    TokenTree::Sequence(
                        delim_span,
                        SequenceRepetition { tts: sequence, separator, kleene, num_captures },
                    )
                }

                // `tree` is followed by an `ident`. This could be `$meta_var` or the `$crate`
                // special metavariable that names the crate of the invocation.
                Some(tokenstream::TokenTree::Token(token, _)) if token.is_ident() => {
                    let (ident, is_raw) = token.ident().unwrap();
                    let span = ident.span.with_lo(dollar_span.lo());
                    if ident.name == kw::Crate && matches!(is_raw, IdentIsRaw::No) {
                        TokenTree::token(token::Ident(kw::DollarCrate, is_raw), span)
                    } else {
                        TokenTree::MetaVar(span, ident)
                    }
                }

                // `tree` is followed by another `$`. This is an escaped `$`.
                Some(&tokenstream::TokenTree::Token(
                    Token { kind: token::Dollar, span: dollar_span2 },
                    _,
                )) => {
                    if parsing_patterns {
                        span_dollar_dollar_or_metavar_in_the_lhs_err(
                            sess,
                            &Token { kind: token::Dollar, span: dollar_span2 },
                        );
                    } else {
                        maybe_emit_macro_metavar_expr_feature(features, sess, dollar_span2);
                    }
                    TokenTree::token(token::Dollar, dollar_span2)
                }

                // `tree` is followed by some other token. This is an error.
                Some(tokenstream::TokenTree::Token(token, _)) => {
                    let msg =
                        format!("expected identifier, found `{}`", pprust::token_to_string(token),);
                    sess.dcx().span_err(token.span, msg);
                    TokenTree::MetaVar(token.span, Ident::dummy())
                }

                // There are no more tokens. Just return the `$` we already have.
                None => TokenTree::token(token::Dollar, dollar_span),
            }
        }

        // `tree` is an arbitrary token. Keep it.
        tokenstream::TokenTree::Token(token, _) => TokenTree::Token(*token),

        // `tree` is the beginning of a delimited set of tokens (e.g., `(` or `{`). We need to
        // descend into the delimited set and further parse it.
        &tokenstream::TokenTree::Delimited(span, spacing, delim, ref tts) => TokenTree::Delimited(
            span,
            spacing,
            Delimited {
                delim,
                tts: parse(tts, parsing_patterns, sess, node_id, features, edition),
            },
        ),
    }
}

/// Takes a token and returns `Some(KleeneOp)` if the token is `+` `*` or `?`. Otherwise, return
/// `None`.
fn kleene_op(token: &Token) -> Option<KleeneOp> {
    match token.kind {
        token::Star => Some(KleeneOp::ZeroOrMore),
        token::Plus => Some(KleeneOp::OneOrMore),
        token::Question => Some(KleeneOp::ZeroOrOne),
        _ => None,
    }
}

/// Parse the next token tree of the input looking for a KleeneOp. Returns
///
/// - Ok(Ok((op, span))) if the next token tree is a KleeneOp
/// - Ok(Err(tok, span)) if the next token tree is a token but not a KleeneOp
/// - Err(span) if the next token tree is not a token
fn parse_kleene_op(
    iter: &mut TokenStreamIter<'_>,
    span: Span,
) -> Result<Result<(KleeneOp, Span), Token>, Span> {
    match iter.next() {
        Some(tokenstream::TokenTree::Token(token, _)) => match kleene_op(token) {
            Some(op) => Ok(Ok((op, token.span))),
            None => Ok(Err(*token)),
        },
        tree => Err(tree.map_or(span, tokenstream::TokenTree::span)),
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
/// This function will take some input iterator `iter` corresponding to `span` and a parsing
/// session `sess`. If the next one (or possibly two) tokens in `iter` correspond to a Kleene
/// operator and separator, then a tuple with `(separator, KleeneOp)` is returned. Otherwise, an
/// error with the appropriate span is emitted to `sess` and a dummy value is returned.
fn parse_sep_and_kleene_op(
    iter: &mut TokenStreamIter<'_>,
    span: Span,
    sess: &Session,
) -> (Option<Token>, KleeneToken) {
    // We basically look at two token trees here, denoted as #1 and #2 below
    let span = match parse_kleene_op(iter, span) {
        // #1 is a `?`, `+`, or `*` KleeneOp
        Ok(Ok((op, span))) => return (None, KleeneToken::new(op, span)),

        // #1 is a separator followed by #2, a KleeneOp
        Ok(Err(token)) => match parse_kleene_op(iter, token.span) {
            // #2 is the `?` Kleene op, which does not take a separator (error)
            Ok(Ok((KleeneOp::ZeroOrOne, span))) => {
                // Error!
                sess.dcx().span_err(
                    token.span,
                    "the `?` macro repetition operator does not take a separator",
                );

                // Return a dummy
                return (None, KleeneToken::new(KleeneOp::ZeroOrMore, span));
            }

            // #2 is a KleeneOp :D
            Ok(Ok((op, span))) => return (Some(token), KleeneToken::new(op, span)),

            // #2 is a random token or not a token at all :(
            Ok(Err(Token { span, .. })) | Err(span) => span,
        },

        // #1 is not a token
        Err(span) => span,
    };

    // If we ever get to this point, we have experienced an "unexpected token" error
    sess.dcx().span_err(span, "expected one of: `*`, `+`, or `?`");

    // Return a dummy
    (None, KleeneToken::new(KleeneOp::ZeroOrMore, span))
}

// `$$` or a meta-variable is the lhs of a macro but shouldn't.
//
// For example, `macro_rules! foo { ( ${len()} ) => {} }`
fn span_dollar_dollar_or_metavar_in_the_lhs_err(sess: &Session, token: &Token) {
    sess.dcx()
        .span_err(token.span, format!("unexpected token: {}", pprust::token_to_string(token)));
    sess.dcx().span_note(
        token.span,
        "`$$` and meta-variable expressions are not allowed inside macro parameter definitions",
    );
}
