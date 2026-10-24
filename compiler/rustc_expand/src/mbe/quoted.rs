use rustc_ast::token::{self, Delimiter, IdentKind, NonterminalKind, Token};
use rustc_ast::tokenstream;
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan};
use rustc_ast_pretty::pprust;
use rustc_errors::Applicability;
use rustc_feature::Features;
use rustc_parse::exp;
use rustc_parse::parser::Parser;
use rustc_session::Session;
use rustc_session::diagnostics::feature_err;
use rustc_span::edition::Edition;
use rustc_span::{Ident, Span, Symbol, kw, sym};

use crate::diagnostics;
use crate::mbe::macro_parser::count_metavar_decls;
use crate::mbe::{Delimited, KleeneOp, KleeneToken, MetaVarExpr, SequenceRepetition, TokenTree};

pub(crate) const VALID_FRAGMENT_NAMES_MSG: &str = "valid fragment specifiers are \
    `ident`, `block`, `stmt`, `expr`, `pat`, `ty`, `lifetime`, `literal`, `path`, \
    `meta`, `tt`, `item` and `vis`, along with `expr_2021` and `pat_param` for edition compatibility";

/// Which part of a macro rule we're parsing
#[derive(Copy, Clone)]
pub(crate) enum RulePart {
    /// The left-hand side, with patterns and metavar definitions with types
    Pattern,
    /// The right-hand side body, with metavar references and metavar expressions
    Body,
}

impl RulePart {
    #[inline(always)]
    fn is_pattern(&self) -> bool {
        matches!(self, Self::Pattern)
    }

    #[inline(always)]
    fn is_body(&self) -> bool {
        matches!(self, Self::Body)
    }
}

/// Takes a `tokenstream::TokenStream` and returns a `Vec<self::TokenTree>`. Specifically, this
/// takes a generic `TokenStream`, such as is used in the rest of the compiler, and returns a
/// collection of `TokenTree` for use in parsing a macro.
///
/// # Parameters
///
/// - `parser`: a parser to read from.
/// - `part`: whether we're parsing the patterns or the body of a macro. Both take roughly the same
///   form _except_ that:
///   - In a pattern, metavars are declared with their "matcher" type. For example `$var:expr` or
///     `$id:ident`. In this example, `expr` and `ident` are "matchers". They are not present in the
///     body of a macro rule -- just in the pattern.
///   - Metavariable expressions are only valid in the "body", not the "pattern".
/// - `sess`: the parsing session. Any errors will be emitted to this session.
/// - `features`: language features so we can do feature gating.
///
/// # Returns
///
/// A collection of `self::TokenTree`. There may also be some errors emitted to `sess`.
fn parse(
    parser: &mut Parser<'_>,
    part: RulePart,
    sess: &Session,
    features: &Features,
    edition: Edition,
) -> Vec<TokenTree> {
    // Will contain the final collection of `self::TokenTree`
    let mut result = Vec::new();

    // For each token tree in `input`, parse the token into a `self::TokenTree`, consuming
    // additional trees if need be.
    while !parser.token.kind.is_close_delim_or_eof() {
        result.push(parse_tree(parser, part, sess, features, edition));
    }
    result
}

/// Takes a `tokenstream::TokenTree` from `parser` and returns a `self::TokenTree`. Like `parse`,
/// but for a single token tree. Emits errors to `sess` if needed.
#[inline]
pub(super) fn parse_one_tt(
    parser: &mut Parser<'_>,
    part: RulePart,
    sess: &Session,
    features: &Features,
    edition: Edition,
) -> TokenTree {
    if parser.token.kind.open_delim().is_none() {
        // Single token; technically an error, but pass through as-is for "recovery".
        let token = parser.token;
        parser.bump();
        return TokenTree::Token(token);
    }

    // Parse the delimited content.
    let (dspan, dspacing, delim, tts) =
        parse_delimited(parser, |parser| parse(parser, part, sess, features, edition));

    TokenTree::Delimited(dspan, dspacing, Delimited { delim, tts })
}

/// Parse delimited tokens.
fn parse_delimited<R>(
    parser: &mut Parser<'_>,
    f: impl FnOnce(&mut Parser<'_>) -> R,
) -> (DelimSpan, DelimSpacing, Delimiter, R) {
    debug_assert!(parser.token.kind.open_delim().is_some());
    let tokenstream::TokenTree::Delimited(dspan, dspacing, delim, _) =
        parser.clone_enclosing_delim()
    else {
        unreachable!()
    };
    parser.bump();
    let inner = (f)(parser);
    debug_assert_eq!(parser.token.kind.close_delim(), Some(delim));
    parser.bump();
    (dspan, dspacing, delim, inner)
}

/// Parse through invisible delimiters, ignoring unused content.
///
/// This only exists for compatibility with previous behavior.
fn badly_parse_invis_delimited<R>(
    parser: &mut Parser<'_>,
    f: impl FnOnce(&mut Parser<'_>) -> R,
) -> R {
    let Some(delim) = parser.token.kind.open_delim().filter(|d| d.skip()) else {
        // No invisible delimiters to skip through.
        return (f)(parser);
    };

    // Skip the opening delimiter.
    parser.bump();

    let result = (f)(parser);

    // Ignore unused content inside the delimiters.
    // FIXME(@bal-e): fail instead of ignoring unused content.
    while !parser.token.kind.is_close_delim_or_eof() {
        parser.parse_token_tree();
    }

    // Skip the closing delimiter.
    debug_assert_eq!(parser.token.kind.close_delim(), Some(delim));
    parser.bump();

    result
}

/// Asks for the `macro_metavar_expr` feature if it is not enabled
fn maybe_emit_macro_metavar_expr_feature(features: &Features, sess: &Session, span: Span) {
    if !features.macro_metavar_expr() {
        let msg = "meta-variable expressions are unstable";
        feature_err(sess, sym::macro_metavar_expr, span, msg).emit();
    }
}

fn maybe_emit_macro_metavar_expr_concat_feature(
    features: &Features,
    sess: &Session,
    span: Span,
    what: Symbol,
) {
    if !features.macro_metavar_expr_concat() {
        let msg = format!("the `{what}` meta-variable expression is unstable");
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
/// - `parser`: The parser to read from.
/// - `part`: same as [parse].
/// - `sess`: the parsing session. Any errors will be emitted to this session.
/// - `features`: language features so we can do feature gating.
fn parse_tree(
    parser: &mut Parser<'_>,
    part: RulePart,
    sess: &Session,
    features: &Features,
    edition: Edition,
) -> TokenTree {
    if !parser.eat(exp!(Dollar)) {
        if parser.token.kind.open_delim().is_none() {
            // `tree` is an arbitrary token. Keep it.
            parser.bump();
            return TokenTree::Token(parser.prev_token);
        } else {
            // `tree` is the beginning of a delimited set of tokens (e.g., `(` or `{`). We need to
            // descend into the delimited set and further parse it.
            let (dspan, dspacing, delim, tts) =
                parse_delimited(parser, |parser| parse(parser, part, sess, features, edition));
            return TokenTree::Delimited(dspan, dspacing, Delimited { delim, tts });
        }
    };

    // `tree` is a `$` token. Look at the next token in `trees`
    let dollar = parser.prev_token;
    // FIXME: Handle `Invisible`-delimited groups in a more systematic way
    // during parsing.
    badly_parse_invis_delimited(parser, |parser| {
        if parser.eat(exp!(OpenParen)) {
            // A repetition sequence in the macro (e.g. `$(pat)*`). Parse the contents of the sequence
            // itself
            let open_span = parser.prev_token.span;
            let sequence = parse(parser, part, sess, features, edition);
            debug_assert_eq!(parser.token.kind, token::CloseParen);
            parser.bump();
            let close_span = parser.prev_token.span;
            let delim_span = DelimSpan { open: open_span, close: close_span };
            // Get the Kleene operator and optional separator
            let (separator, kleene) = parse_sep_and_kleene_op(parser, delim_span.entire(), sess);
            // Count the number of captured "names" (i.e., named metavars)
            let num_captures = if part.is_pattern() { count_metavar_decls(&sequence) } else { 0 };
            TokenTree::Sequence(
                delim_span,
                SequenceRepetition { tts: sequence, separator, kleene, num_captures },
            )
        } else if part.is_body() && parser.token == token::OpenBrace {
            // The delimiter is `{`. This indicates the beginning
            // of a meta-variable expression (e.g. `${count(ident)}`).
            // Try to parse the meta-variable expression.
            let tokenstream::TokenTree::Delimited(delim_span, _, _, tts) =
                parser.parse_token_tree()
            else {
                unreachable!()
            };
            match MetaVarExpr::parse(&tts, delim_span.entire(), &sess.psess) {
                Err(err) => {
                    err.emit();
                    // Returns early the same read `$` to avoid spanning
                    // unrelated diagnostics that could be performed afterwards
                    return TokenTree::Token(dollar);
                }
                Ok(elem) => {
                    if matches!(elem, MetaVarExpr::ConcatIdent(_)) {
                        maybe_emit_macro_metavar_expr_concat_feature(
                            features,
                            sess,
                            delim_span.entire(),
                            sym::concat,
                        );
                    } else if matches!(elem, MetaVarExpr::ConcatStr(_)) {
                        maybe_emit_macro_metavar_expr_concat_feature(
                            features,
                            sess,
                            delim_span.entire(),
                            sym::concat_str,
                        );
                    } else {
                        maybe_emit_macro_metavar_expr_feature(features, sess, delim_span.entire());
                    }
                    return TokenTree::MetaVarExpr(delim_span, elem);
                }
            }
        } else if parser.token.kind.open_delim().is_some() {
            // Disallowed kind of delimiters.
            let (dspan, _, delim, tts) =
                parse_delimited(parser, |parser| parse(parser, part, sess, features, edition));
            if part.is_pattern() {
                span_dollar_dollar_or_metavar_in_the_lhs_err(
                    sess,
                    &Token { kind: delim.as_open_token_kind(), span: dspan.entire() },
                );
            } else {
                let token = pprust::token_kind_to_string(&delim.as_open_token_kind());
                sess.dcx()
                    .emit_err(diagnostics::ExpectedParenOrBrace { span: dspan.entire(), token });
            }
            let (separator, kleene) = parse_sep_and_kleene_op(parser, dspan.entire(), sess);
            let num_captures = if part.is_pattern() { count_metavar_decls(&tts) } else { 0 };

            TokenTree::Sequence(dspan, SequenceRepetition { tts, separator, kleene, num_captures })
        } else if parser.token.is_ident() {
            parse_dollar_ident(parser, sess, edition, part, dollar.span)
        } else if parser.eat(exp!(Dollar)) {
            // `tree` is followed by another `$`. This is an escaped `$`.
            let dollar_span2 = parser.prev_token.span;
            if part.is_pattern() {
                span_dollar_dollar_or_metavar_in_the_lhs_err(sess, &parser.prev_token);
            } else {
                maybe_emit_macro_metavar_expr_feature(features, sess, dollar_span2);
            }
            TokenTree::Token(parser.prev_token)
        } else if !parser.token.kind.is_close_delim_or_eof() {
            // `$` is followed by some other token. This is an error.
            let token = parser.token;
            parser.bump();
            let msg = format!("expected identifier, found `{}`", pprust::token_to_string(&token));
            sess.dcx().span_err(token.span, msg);
            // Return a dummy meta-variable declaration/reference.
            let span = token.span;
            let ident = Ident::dummy();
            match part {
                RulePart::Pattern => parse_metavar_decl(parser, sess, edition, span, ident),
                RulePart::Body => TokenTree::MetaVar(span, ident),
            }
        } else {
            // There are no more tokens. Just return the `$` we already have.
            TokenTree::Token(dollar)
        }
    })
}

/// Parses `$ident`, i.e. a meta-variable.
///
/// The parser must be located after `$`.
fn parse_dollar_ident(
    parser: &mut Parser<'_>,
    sess: &Session,
    edition: Edition,
    part: RulePart,
    dollar_span: Span,
) -> TokenTree {
    let (ident, kind) = parser.token.ident().unwrap();
    let span = ident.span.with_lo(dollar_span.lo());
    parser.bump();

    if let kw::Crate = ident.name
        && let IdentKind::Normal | IdentKind::ForcedKeyword = kind
    {
        // This is the special `$crate` meta-variable.
        return TokenTree::Token(Token::new(token::Ident(kw::DollarCrate, kind), span));
    }

    // This is a regular meta-variable.
    match part {
        // A meta-variable declaration.
        RulePart::Pattern => parse_metavar_decl(parser, sess, edition, span, ident),

        // A meta-variable reference.
        RulePart::Body => TokenTree::MetaVar(span, ident),
    }
}

/// Parses `$metavar:kind`, i.e. a meta-variable declaration.
///
/// The parser must be located after `$metavar`.
fn parse_metavar_decl(
    parser: &mut Parser<'_>,
    sess: &Session,
    edition: Edition,
    metavar_span: Span,
    ident: Ident,
) -> TokenTree {
    let fallback_metavar_decl =
        |span| TokenTree::MetaVarDecl { span, name: ident, kind: NonterminalKind::TT };

    // Emit a missing-fragment diagnostic and return a `TokenTree` fallback so parsing can
    // continue.
    let missing_fragment_specifier = |span, add_span| {
        sess.dcx().emit_err(diagnostics::MissingFragmentSpecifier {
            span,
            add_span,
            valid: VALID_FRAGMENT_NAMES_MSG,
        });
        fallback_metavar_decl(span)
    };

    if !parser.eat(exp!(Colon)) {
        // Whether it's none or some other tree, it doesn't belong to
        // the current meta variable, returning the original span.
        return missing_fragment_specifier(metavar_span, metavar_span.shrink_to_hi());
    }

    let colon = parser.prev_token;
    let Some((fragment, _)) = parser.token.ident() else {
        // No identifier for the fragment specifier;
        if parser.token.kind == token::Dollar && parser.look_ahead(1, |next| next.is_ident()) {
            let mut err =
                sess.dcx().struct_span_err(parser.token.span, "missing fragment specifier");
            err.note("fragment specifiers must be provided");
            err.help(VALID_FRAGMENT_NAMES_MSG);
            err.span_suggestion_verbose(
                parser.token.span,
                "fragment specifiers should not be prefixed with `$`",
                "",
                Applicability::MaybeIncorrect,
            );
            err.emit();
            parser.bump(); // past `$`
            return fallback_metavar_decl(parser.prev_token.span);
        } else if parser.token.kind.open_delim().is_some() {
            let _ = parser.parse_token_tree();
            return missing_fragment_specifier(
                colon.span.with_lo(metavar_span.lo()),
                colon.span.shrink_to_hi(),
            );
        } else {
            let _ = parser.parse_token_tree();
            return missing_fragment_specifier(colon.span, colon.span.shrink_to_hi());
        }
    };

    parser.bump();
    let span = parser.prev_token.span.with_lo(metavar_span.lo());
    let edition = || {
        // FIXME(#85708) - once we properly decode a foreign
        // crate's `SyntaxContext::root`, then we can replace
        // this with just `span.edition()`. A
        // `SyntaxContext::root()` from the current crate will
        // have the edition of the current crate, and a
        // `SyntaxContext::root()` from a foreign crate will
        // have the edition of that crate (which we manually
        // retrieve via the `edition` parameter).
        if !span.from_expansion() { edition } else { span.edition() }
    };
    let kind = NonterminalKind::from_symbol(fragment.name, edition).unwrap_or_else(|| {
        sess.dcx().emit_err(diagnostics::InvalidFragmentSpecifier {
            span,
            fragment,
            help: VALID_FRAGMENT_NAMES_MSG,
        });
        NonterminalKind::TT
    });
    TokenTree::MetaVarDecl { span, name: ident, kind }
}

/// Parses a Kleene operator, possibly with a separator.
///
/// Parses `,*` in `$(a),*`. `parser` should be located at `,`; one or two token trees will be
/// consumed. On failure, an error is emitted and a dummy value is returned.
fn parse_sep_and_kleene_op(
    parser: &mut Parser<'_>,
    span: Span,
    sess: &Session,
) -> (Option<Token>, KleeneToken) {
    /// Identify a Kleene op.
    fn kleene_op(token: &Token) -> Option<KleeneOp> {
        match token.kind {
            token::Star => Some(KleeneOp::ZeroOrMore),
            token::Plus => Some(KleeneOp::OneOrMore),
            token::Question => Some(KleeneOp::ZeroOrOne),
            _ => None,
        }
    }

    if let Some(op) = kleene_op(&parser.token) {
        // Kleene op without a separator
        let token = parser.token;
        parser.bump();
        (None, KleeneToken::new(op, token.span))
    } else if !parser.token.kind.is_delim()
        && let Some(op) = parser.look_ahead(1, kleene_op)
    {
        // Separator followed by a Kleene op
        let token = parser.token;
        parser.bump();
        let kleene = parser.token;
        parser.bump();

        if let KleeneOp::ZeroOrOne = op {
            sess.dcx().span_err(
                token.span,
                "the `?` macro repetition operator does not take a separator",
            );

            // Recover by ignoring the separator.
            return (None, KleeneToken::new(KleeneOp::ZeroOrMore, kleene.span));
        }

        (Some(token), KleeneToken::new(op, kleene.span))
    } else {
        // No Kleene op found -- error

        let span = if parser.token.kind.is_close_delim_or_eof() {
            span
        } else if parser.token.kind.open_delim().is_some() {
            parser.parse_token_tree().span()
        } else {
            parser.bump(); // skip potential separator
            parser.parse_token_tree().span()
        };

        sess.dcx().span_err(span, "expected one of: `*`, `+`, or `?`");

        // Recover using a dummy Kleene op.
        (None, KleeneToken::new(KleeneOp::ZeroOrMore, span))
    }
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
