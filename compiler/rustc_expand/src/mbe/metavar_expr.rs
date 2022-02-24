use std::fmt::Display;
use std::str::FromStr;

use rustc_ast::token::{self, Lit};
use rustc_ast::tokenstream;
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, PResult};
use rustc_session::parse::ParseSess;

use rustc_span::symbol::Ident;
use rustc_span::Span;

/// A meta-variable expression, for expansions based on properties of meta-variables.
#[derive(Debug, Clone, PartialEq, Encodable, Decodable)]
crate enum MetaVarExpr {
    /// The number of repetitions of an identifier, optionally limited to a number
    /// of outer-most repetition depths. If the depth limit is `None` then the depth is unlimited.
    Count(Ident, Option<usize>),

    /// Ignore a meta-variable for repetition without expansion.
    Ignore(Ident),

    /// The index of the repetition at a particular depth, where 0 is the inner-most
    /// repetition. The `usize` is the depth.
    Index(usize),

    /// The length of the repetition at a particular depth, where 0 is the inner-most
    /// repetition. The `usize` is the depth.
    Length(usize),
}

impl MetaVarExpr {
    /// Attempt to parse a meta-variable expression from a token stream.
    crate fn parse<'sess>(
        input: &tokenstream::TokenStream,
        sess: &'sess ParseSess,
    ) -> PResult<'sess, MetaVarExpr> {
        let mut tts = input.trees();
        match tts.next() {
            Some(tokenstream::TokenTree::Token(token)) if let Some((ident, false)) = token.ident() => {
                let Some(tokenstream::TokenTree::Delimited(_, token::Paren, args)) = tts.next() else {
                    let msg = "meta-variable expression paramter must be wrapped in parentheses";
                    return Err(sess.span_diagnostic.struct_span_err(ident.span, msg));
                };
                let mut iter = args.trees();
                let rslt = match &*ident.as_str() {
                    "count" => parse_count(&mut iter, sess, ident.span)?,
                    "ignore" => MetaVarExpr::Ignore(parse_ident(&mut iter, sess, ident.span)?),
                    "index" => MetaVarExpr::Index(parse_depth(&mut iter, sess, ident.span)?),
                    "length" => MetaVarExpr::Length(parse_depth(&mut iter, sess, ident.span)?),
                    _ => {
                        let msg = "unrecognised meta-variable expression. Supported expressions \
                        are count, ignore, index and length";
                        return Err(sess.span_diagnostic.struct_span_err(ident.span, msg));
                    }
                };
                if let Some(arg) = iter.next() {
                    let msg = "unexpected meta-variable expression argument";
                    return Err(sess.span_diagnostic.struct_span_err(arg.span(), msg));
                }
                Ok(rslt)
            }
            Some(tokenstream::TokenTree::Token(token)) => {
                return Err(sess.span_diagnostic.struct_span_err(
                    token.span,
                    &format!(
                        "expected meta-variable expression, found `{}`",
                        pprust::token_to_string(&token),
                    ),
                ));
            }
            _ => return Err(sess.span_diagnostic.struct_err("expected meta-variable expression"))
        }
    }

    crate fn ident(&self) -> Option<&Ident> {
        match self {
            MetaVarExpr::Count(ident, _) | MetaVarExpr::Ignore(ident) => Some(&ident),
            MetaVarExpr::Index(..) | MetaVarExpr::Length(..) => None,
        }
    }
}

/// Tries to convert a literal to an arbitrary type
fn convert_literal<T>(lit: Lit, sess: &ParseSess, span: Span) -> PResult<'_, T>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    if lit.suffix.is_some() {
        let msg = "literal suffixes are not supported in meta-variable expressions";
        return Err(sess.span_diagnostic.struct_span_err(span, msg));
    }
    lit.symbol.as_str().parse::<T>().map_err(|e| {
        sess.span_diagnostic.struct_span_err(
            span,
            &format!("failed to parse meta-variable expression argument: {}", e),
        )
    })
}

/// Parse a meta-variable `count` expression: `count(ident[, depth])`
fn parse_count<'sess>(
    iter: &mut tokenstream::Cursor,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, MetaVarExpr> {
    let ident = parse_ident(iter, sess, span)?;
    let depth = if try_eat_comma(iter) { Some(parse_depth(iter, sess, span)?) } else { None };
    Ok(MetaVarExpr::Count(ident, depth))
}

/// Parses the depth used by index(depth) and length(depth).
fn parse_depth<'sess>(
    iter: &mut tokenstream::Cursor,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, usize> {
    let Some(tt) = iter.next() else { return Ok(0) };
    let tokenstream::TokenTree::Token(token::Token {
        kind: token::TokenKind::Literal(lit),
        span: literal_span,
    }) = tt else {
        return Err(sess.span_diagnostic.struct_span_err(
            span,
            "meta-expression depth must be a literal"
        ));
    };
    convert_literal::<usize>(lit, sess, literal_span)
}

/// Parses an generic ident
fn parse_ident<'sess>(
    iter: &mut tokenstream::Cursor,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, Ident> {
    let err_fn =
        || sess.span_diagnostic.struct_span_err(span, "could not find an expected `ident` element");
    if let Some(tt) = iter.next() {
        match tt {
            tokenstream::TokenTree::Token(token) => {
                if let Some((elem, false)) = token.ident() {
                    return Ok(elem);
                }
                let mut err = err_fn();
                err.span_suggestion(
                    token.span,
                    &format!("Try removing `{}`", pprust::token_to_string(&token)),
                    <_>::default(),
                    Applicability::MaybeIncorrect,
                );
                return Err(err);
            }
            tokenstream::TokenTree::Delimited(delim_span, _, _) => {
                let mut err = err_fn();
                err.span_suggestion(
                    delim_span.entire(),
                    "Try removing the delimiter",
                    <_>::default(),
                    Applicability::MaybeIncorrect,
                );
                return Err(err);
            }
        }
    }
    Err(err_fn())
}

/// Tries to move the iterator forward returning `true` if there is a comma. If not, then the
/// iterator is not modified and the result is `false`.
fn try_eat_comma(iter: &mut tokenstream::Cursor) -> bool {
    if let Some(tokenstream::TokenTree::Token(token::Token { kind: token::Comma, .. })) =
        iter.look_ahead(0)
    {
        let _ = iter.next();
        return true;
    }
    false
}
