use rustc_ast::token::{self, Delimiter, IdentIsRaw, Lit, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenStreamIter, TokenTree};
use rustc_ast::{LitIntType, LitKind};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, PResult};
use rustc_macros::{Decodable, Encodable};
use rustc_session::parse::ParseSess;
use rustc_span::{Ident, Span, Symbol};

pub(crate) const RAW_IDENT_ERR: &str = "`${concat(..)}` currently does not support raw identifiers";
pub(crate) const UNSUPPORTED_CONCAT_ELEM_ERR: &str = "expected identifier or string literal";

/// A meta-variable expression, for expansions based on properties of meta-variables.
#[derive(Debug, PartialEq, Encodable, Decodable)]
pub(crate) enum MetaVarExpr {
    /// Unification of two or more identifiers.
    Concat(Box<[MetaVarExprConcatElem]>),

    /// The number of repetitions of an identifier.
    Count(Ident, usize),

    /// Ignore a meta-variable for repetition without expansion.
    Ignore(Ident),

    /// The index of the repetition at a particular depth, where 0 is the innermost
    /// repetition. The `usize` is the depth.
    Index(usize),

    /// The length of the repetition at a particular depth, where 0 is the innermost
    /// repetition. The `usize` is the depth.
    Len(usize),
}

impl MetaVarExpr {
    /// Attempt to parse a meta-variable expression from a token stream.
    pub(crate) fn parse<'psess>(
        input: &TokenStream,
        outer_span: Span,
        psess: &'psess ParseSess,
    ) -> PResult<'psess, MetaVarExpr> {
        let mut iter = input.iter();
        let ident = parse_ident(&mut iter, psess, outer_span)?;
        let Some(TokenTree::Delimited(.., Delimiter::Parenthesis, args)) = iter.next() else {
            let msg = "meta-variable expression parameter must be wrapped in parentheses";
            return Err(psess.dcx().struct_span_err(ident.span, msg));
        };
        check_trailing_token(&mut iter, psess)?;
        let mut iter = args.iter();
        let rslt = match ident.as_str() {
            "concat" => parse_concat(&mut iter, psess, outer_span, ident.span)?,
            "count" => parse_count(&mut iter, psess, ident.span)?,
            "ignore" => {
                eat_dollar(&mut iter, psess, ident.span)?;
                MetaVarExpr::Ignore(parse_ident(&mut iter, psess, ident.span)?)
            }
            "index" => MetaVarExpr::Index(parse_depth(&mut iter, psess, ident.span)?),
            "len" => MetaVarExpr::Len(parse_depth(&mut iter, psess, ident.span)?),
            _ => {
                let err_msg = "unrecognized meta-variable expression";
                let mut err = psess.dcx().struct_span_err(ident.span, err_msg);
                err.span_suggestion(
                    ident.span,
                    "supported expressions are count, ignore, index and len",
                    "",
                    Applicability::MachineApplicable,
                );
                return Err(err);
            }
        };
        check_trailing_token(&mut iter, psess)?;
        Ok(rslt)
    }

    pub(crate) fn for_each_metavar<A>(&self, mut aux: A, mut cb: impl FnMut(A, &Ident) -> A) -> A {
        match self {
            MetaVarExpr::Concat(elems) => {
                for elem in elems {
                    if let MetaVarExprConcatElem::Var(ident) = elem {
                        aux = cb(aux, ident)
                    }
                }
                aux
            }
            MetaVarExpr::Count(ident, _) | MetaVarExpr::Ignore(ident) => cb(aux, ident),
            MetaVarExpr::Index(..) | MetaVarExpr::Len(..) => aux,
        }
    }
}

// Checks if there are any remaining tokens. For example, `${ignore(ident ... a b c ...)}`
fn check_trailing_token<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
) -> PResult<'psess, ()> {
    if let Some(tt) = iter.next() {
        let mut diag = psess
            .dcx()
            .struct_span_err(tt.span(), format!("unexpected token: {}", pprust::tt_to_string(tt)));
        diag.span_note(tt.span(), "meta-variable expression must not have trailing tokens");
        Err(diag)
    } else {
        Ok(())
    }
}

/// Indicates what is placed in a `concat` parameter. For example, literals
/// (`${concat("foo", "bar")}`) or adhoc identifiers (`${concat(foo, bar)}`).
#[derive(Debug, Decodable, Encodable, PartialEq)]
pub(crate) enum MetaVarExprConcatElem {
    /// Identifier WITHOUT a preceding dollar sign, which means that this identifier should be
    /// interpreted as a literal.
    Ident(Ident),
    /// For example, a number or a string.
    Literal(Symbol),
    /// Identifier WITH a preceding dollar sign, which means that this identifier should be
    /// expanded and interpreted as a variable.
    Var(Ident),
}

/// Parse a meta-variable `concat` expression: `concat($metavar, ident, ...)`.
fn parse_concat<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    outer_span: Span,
    expr_ident_span: Span,
) -> PResult<'psess, MetaVarExpr> {
    let mut result = Vec::new();
    loop {
        let is_var = try_eat_dollar(iter);
        let token = parse_token(iter, psess, outer_span)?;
        let element = if is_var {
            MetaVarExprConcatElem::Var(parse_ident_from_token(psess, token)?)
        } else if let TokenKind::Literal(Lit { kind: token::LitKind::Str, symbol, suffix: None }) =
            token.kind
        {
            MetaVarExprConcatElem::Literal(symbol)
        } else {
            match parse_ident_from_token(psess, token) {
                Err(err) => {
                    err.cancel();
                    return Err(psess
                        .dcx()
                        .struct_span_err(token.span, UNSUPPORTED_CONCAT_ELEM_ERR));
                }
                Ok(elem) => MetaVarExprConcatElem::Ident(elem),
            }
        };
        result.push(element);
        if iter.peek().is_none() {
            break;
        }
        if !try_eat_comma(iter) {
            return Err(psess.dcx().struct_span_err(outer_span, "expected comma"));
        }
    }
    if result.len() < 2 {
        return Err(psess
            .dcx()
            .struct_span_err(expr_ident_span, "`concat` must have at least two elements"));
    }
    Ok(MetaVarExpr::Concat(result.into()))
}

/// Parse a meta-variable `count` expression: `count(ident[, depth])`
fn parse_count<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, MetaVarExpr> {
    eat_dollar(iter, psess, span)?;
    let ident = parse_ident(iter, psess, span)?;
    let depth = if try_eat_comma(iter) {
        if iter.peek().is_none() {
            return Err(psess.dcx().struct_span_err(
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

/// Parses the depth used by index(depth) and len(depth).
fn parse_depth<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, usize> {
    let Some(tt) = iter.next() else { return Ok(0) };
    let TokenTree::Token(Token { kind: TokenKind::Literal(lit), .. }, _) = tt else {
        return Err(psess
            .dcx()
            .struct_span_err(span, "meta-variable expression depth must be a literal"));
    };
    if let Ok(lit_kind) = LitKind::from_token_lit(*lit)
        && let LitKind::Int(n_u128, LitIntType::Unsuffixed) = lit_kind
        && let Ok(n_usize) = usize::try_from(n_u128.get())
    {
        Ok(n_usize)
    } else {
        let msg = "only unsuffixes integer literals are supported in meta-variable expressions";
        Err(psess.dcx().struct_span_err(span, msg))
    }
}

/// Parses an generic ident
fn parse_ident<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    fallback_span: Span,
) -> PResult<'psess, Ident> {
    let token = parse_token(iter, psess, fallback_span)?;
    parse_ident_from_token(psess, token)
}

fn parse_ident_from_token<'psess>(
    psess: &'psess ParseSess,
    token: &Token,
) -> PResult<'psess, Ident> {
    if let Some((elem, is_raw)) = token.ident() {
        if let IdentIsRaw::Yes = is_raw {
            return Err(psess.dcx().struct_span_err(elem.span, RAW_IDENT_ERR));
        }
        return Ok(elem);
    }
    let token_str = pprust::token_to_string(token);
    let mut err = psess
        .dcx()
        .struct_span_err(token.span, format!("expected identifier, found `{token_str}`"));
    err.span_suggestion(
        token.span,
        format!("try removing `{token_str}`"),
        "",
        Applicability::MaybeIncorrect,
    );
    Err(err)
}

fn parse_token<'psess, 't>(
    iter: &mut TokenStreamIter<'t>,
    psess: &'psess ParseSess,
    fallback_span: Span,
) -> PResult<'psess, &'t Token> {
    let Some(tt) = iter.next() else {
        return Err(psess.dcx().struct_span_err(fallback_span, UNSUPPORTED_CONCAT_ELEM_ERR));
    };
    let TokenTree::Token(token, _) = tt else {
        return Err(psess.dcx().struct_span_err(tt.span(), UNSUPPORTED_CONCAT_ELEM_ERR));
    };
    Ok(token)
}

/// Tries to move the iterator forward returning `true` if there is a comma. If not, then the
/// iterator is not modified and the result is `false`.
fn try_eat_comma(iter: &mut TokenStreamIter<'_>) -> bool {
    if let Some(TokenTree::Token(Token { kind: token::Comma, .. }, _)) = iter.peek() {
        let _ = iter.next();
        return true;
    }
    false
}

/// Tries to move the iterator forward returning `true` if there is a dollar sign. If not, then the
/// iterator is not modified and the result is `false`.
fn try_eat_dollar(iter: &mut TokenStreamIter<'_>) -> bool {
    if let Some(TokenTree::Token(Token { kind: token::Dollar, .. }, _)) = iter.peek() {
        let _ = iter.next();
        return true;
    }
    false
}

/// Expects that the next item is a dollar sign.
fn eat_dollar<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, ()> {
    if try_eat_dollar(iter) {
        return Ok(());
    }
    Err(psess.dcx().struct_span_err(
        span,
        "meta-variables within meta-variable expressions must be referenced using a dollar sign",
    ))
}
