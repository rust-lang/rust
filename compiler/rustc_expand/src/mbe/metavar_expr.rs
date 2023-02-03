use rustc_ast::token::{self, Delimiter};
use rustc_ast::tokenstream::{RefTokenTreeCursor, TokenStream, TokenTree};
use rustc_ast::{LitIntType, LitKind};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::symbol::Ident;
use rustc_span::Span;

/// A meta-variable expression, for expansions based on properties of meta-variables.
#[derive(Debug, Clone, PartialEq, Encodable, Decodable)]
pub(crate) enum MetaVarExpr {
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
    pub(crate) fn parse<'sess>(
        input: &TokenStream,
        outer_span: Span,
        sess: &'sess ParseSess,
    ) -> PResult<'sess, MetaVarExpr> {
        let mut tts = input.trees();
        let ident = parse_ident(&mut tts, sess, outer_span)?;
        let Some(TokenTree::Delimited(_, Delimiter::Parenthesis, args)) = tts.next() else {
            let msg = "meta-variable expression parameter must be wrapped in parentheses";
            return Err(sess.span_diagnostic.struct_span_err(ident.span, msg));
        };
        check_trailing_token(&mut tts, sess)?;
        let mut iter = args.trees();
        let rslt = match &*ident.as_str() {
            "count" => parse_count(&mut iter, sess, ident.span)?,
            "ignore" => MetaVarExpr::Ignore(parse_ident(&mut iter, sess, ident.span)?),
            "index" => MetaVarExpr::Index(parse_depth(&mut iter, sess, ident.span)?),
            "length" => MetaVarExpr::Length(parse_depth(&mut iter, sess, ident.span)?),
            _ => {
                let err_msg = "unrecognized meta-variable expression";
                let mut err = sess.span_diagnostic.struct_span_err(ident.span, err_msg);
                err.span_suggestion(
                    ident.span,
                    "supported expressions are count, ignore, index and length",
                    "",
                    Applicability::MachineApplicable,
                );
                return Err(err);
            }
        };
        check_trailing_token(&mut iter, sess)?;
        Ok(rslt)
    }

    pub(crate) fn ident(&self) -> Option<Ident> {
        match *self {
            MetaVarExpr::Count(ident, _) | MetaVarExpr::Ignore(ident) => Some(ident),
            MetaVarExpr::Index(..) | MetaVarExpr::Length(..) => None,
        }
    }
}

// Checks if there are any remaining tokens. For example, `${ignore(ident ... a b c ...)}`
fn check_trailing_token<'sess>(
    iter: &mut RefTokenTreeCursor<'_>,
    sess: &'sess ParseSess,
) -> PResult<'sess, ()> {
    if let Some(tt) = iter.next() {
        let mut diag = sess
            .span_diagnostic
            .struct_span_err(tt.span(), &format!("unexpected token: {}", pprust::tt_to_string(tt)));
        diag.span_note(tt.span(), "meta-variable expression must not have trailing tokens");
        Err(diag)
    } else {
        Ok(())
    }
}

/// Parse a meta-variable `count` expression: `count(ident[, depth])`
fn parse_count<'sess>(
    iter: &mut RefTokenTreeCursor<'_>,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, MetaVarExpr> {
    let ident = parse_ident(iter, sess, span)?;
    let depth = if try_eat_comma(iter) { Some(parse_depth(iter, sess, span)?) } else { None };
    Ok(MetaVarExpr::Count(ident, depth))
}

/// Parses the depth used by index(depth) and length(depth).
fn parse_depth<'sess>(
    iter: &mut RefTokenTreeCursor<'_>,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, usize> {
    let Some(tt) = iter.next() else { return Ok(0) };
    let TokenTree::Token(token::Token {
        kind: token::TokenKind::Literal(lit), ..
    }, _) = tt else {
        return Err(sess.span_diagnostic.struct_span_err(
            span,
            "meta-variable expression depth must be a literal"
        ));
    };
    if let Ok(lit_kind) = LitKind::from_token_lit(*lit)
        && let LitKind::Int(n_u128, LitIntType::Unsuffixed) = lit_kind
        && let Ok(n_usize) = usize::try_from(n_u128)
    {
        Ok(n_usize)
    }
    else {
        let msg = "only unsuffixes integer literals are supported in meta-variable expressions";
        Err(sess.span_diagnostic.struct_span_err(span, msg))
    }
}

/// Parses an generic ident
fn parse_ident<'sess>(
    iter: &mut RefTokenTreeCursor<'_>,
    sess: &'sess ParseSess,
    span: Span,
) -> PResult<'sess, Ident> {
    if let Some(tt) = iter.next() && let TokenTree::Token(token, _) = tt {
        if let Some((elem, false)) = token.ident() {
            return Ok(elem);
        }
        let token_str = pprust::token_to_string(token);
        let mut err = sess.span_diagnostic.struct_span_err(
            span,
            &format!("expected identifier, found `{}`", &token_str)
        );
        err.span_suggestion(
            token.span,
            &format!("try removing `{}`", &token_str),
            "",
            Applicability::MaybeIncorrect,
        );
        return Err(err);
    }
    Err(sess.span_diagnostic.struct_span_err(span, "expected identifier"))
}

/// Tries to move the iterator forward returning `true` if there is a comma. If not, then the
/// iterator is not modified and the result is `false`.
fn try_eat_comma(iter: &mut RefTokenTreeCursor<'_>) -> bool {
    if let Some(TokenTree::Token(token::Token { kind: token::Comma, .. }, _)) = iter.look_ahead(0) {
        let _ = iter.next();
        return true;
    }
    false
}
