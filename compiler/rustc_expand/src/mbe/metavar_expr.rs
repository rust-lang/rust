use rustc_ast::token::{self, Delimiter, IdentIsRaw};
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
    /// The number of repetitions of an identifier.
    Count(Ident, usize),

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
    pub(crate) fn parse<'psess>(
        input: &TokenStream,
        outer_span: Span,
        psess: &'psess ParseSess,
    ) -> PResult<'psess, MetaVarExpr> {
        let mut tts = input.trees();
        let ident = parse_ident(&mut tts, psess, outer_span)?;
        let Some(TokenTree::Delimited(.., Delimiter::Parenthesis, args)) = tts.next() else {
            let msg = "meta-variable expression parameter must be wrapped in parentheses";
            return Err(psess.dcx.struct_span_err(ident.span, msg));
        };
        check_trailing_token(&mut tts, psess)?;
        let mut iter = args.trees();
        let rslt = match ident.as_str() {
            "count" => parse_count(&mut iter, psess, ident.span)?,
            "ignore" => {
                eat_dollar(&mut iter, psess, ident.span)?;
                MetaVarExpr::Ignore(parse_ident(&mut iter, psess, ident.span)?)
            }
            "index" => MetaVarExpr::Index(parse_depth(&mut iter, psess, ident.span)?),
            "length" => MetaVarExpr::Length(parse_depth(&mut iter, psess, ident.span)?),
            _ => {
                let err_msg = "unrecognized meta-variable expression";
                let mut err = psess.dcx.struct_span_err(ident.span, err_msg);
                err.span_suggestion(
                    ident.span,
                    "supported expressions are count, ignore, index and length",
                    "",
                    Applicability::MachineApplicable,
                );
                return Err(err);
            }
        };
        check_trailing_token(&mut iter, psess)?;
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
fn check_trailing_token<'psess>(
    iter: &mut RefTokenTreeCursor<'_>,
    psess: &'psess ParseSess,
) -> PResult<'psess, ()> {
    if let Some(tt) = iter.next() {
        let mut diag = psess
            .dcx
            .struct_span_err(tt.span(), format!("unexpected token: {}", pprust::tt_to_string(tt)));
        diag.span_note(tt.span(), "meta-variable expression must not have trailing tokens");
        Err(diag)
    } else {
        Ok(())
    }
}

/// Parse a meta-variable `count` expression: `count(ident[, depth])`
fn parse_count<'psess>(
    iter: &mut RefTokenTreeCursor<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, MetaVarExpr> {
    eat_dollar(iter, psess, span)?;
    let ident = parse_ident(iter, psess, span)?;
    let depth = if try_eat_comma(iter) {
        if iter.look_ahead(0).is_none() {
            return Err(psess.dcx.struct_span_err(
                span,
                "`count` followed by a comma must have an associated index indicating its depth",
            ));
        }
        parse_depth(iter, psess, span)?
    } else {
        0
    };
    Ok(MetaVarExpr::Count(ident, depth))
}

/// Parses the depth used by index(depth) and length(depth).
fn parse_depth<'psess>(
    iter: &mut RefTokenTreeCursor<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, usize> {
    let Some(tt) = iter.next() else { return Ok(0) };
    let TokenTree::Token(token::Token { kind: token::TokenKind::Literal(lit), .. }, _) = tt else {
        return Err(psess
            .dcx
            .struct_span_err(span, "meta-variable expression depth must be a literal"));
    };
    if let Ok(lit_kind) = LitKind::from_token_lit(*lit)
        && let LitKind::Int(n_u128, LitIntType::Unsuffixed) = lit_kind
        && let Ok(n_usize) = usize::try_from(n_u128.get())
    {
        Ok(n_usize)
    } else {
        let msg = "only unsuffixes integer literals are supported in meta-variable expressions";
        Err(psess.dcx.struct_span_err(span, msg))
    }
}

/// Parses an generic ident
fn parse_ident<'psess>(
    iter: &mut RefTokenTreeCursor<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, Ident> {
    if let Some(tt) = iter.next()
        && let TokenTree::Token(token, _) = tt
    {
        if let Some((elem, IdentIsRaw::No)) = token.ident() {
            return Ok(elem);
        }
        let token_str = pprust::token_to_string(token);
        let mut err =
            psess.dcx.struct_span_err(span, format!("expected identifier, found `{}`", &token_str));
        err.span_suggestion(
            token.span,
            format!("try removing `{}`", &token_str),
            "",
            Applicability::MaybeIncorrect,
        );
        return Err(err);
    }
    Err(psess.dcx.struct_span_err(span, "expected identifier"))
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

/// Expects that the next item is a dollar sign.
fn eat_dollar<'psess>(
    iter: &mut RefTokenTreeCursor<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, ()> {
    if let Some(TokenTree::Token(token::Token { kind: token::Dollar, .. }, _)) = iter.look_ahead(0)
    {
        let _ = iter.next();
        return Ok(());
    }
    Err(psess.dcx.struct_span_err(
        span,
        "meta-variables within meta-variable expressions must be referenced using a dollar sign",
    ))
}
