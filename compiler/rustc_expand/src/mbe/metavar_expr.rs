use rustc_ast::token::{self, Delimiter, IdentIsRaw, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenStreamIter, TokenTree};
use rustc_ast::{self as ast, LitIntType, LitKind};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_lexer::is_id_continue;
use rustc_macros::{Decodable, Encodable};
use rustc_session::errors::create_lit_error;
use rustc_session::parse::ParseSess;
use rustc_span::{Ident, Span, Symbol};

use crate::errors::{self, MveConcatInvalidReason, MveExpectedIdentContext};

pub(crate) const RAW_IDENT_ERR: &str = "`${concat(..)}` currently does not support raw identifiers";
pub(crate) const VALID_EXPR_CONCAT_TYPES: &str =
    "metavariables, identifiers, string literals, and integer literals";

/// Argument specification for a metavariable expression
#[derive(Clone, Copy)]
enum ArgSpec {
    /// Any number of args
    Any,
    /// Between n and m args (inclusive)
    Between(usize, usize),
    /// Exactly n args
    Exact(usize),
}

/// Map of `(name, max_arg_count, variable_count)`.
const EXPR_NAME_ARG_MAP: &[(&str, ArgSpec)] = &[
    ("concat", ArgSpec::Any),
    ("count", ArgSpec::Between(1, 2)),
    ("ignore", ArgSpec::Exact(1)),
    ("index", ArgSpec::Between(0, 1)),
    ("len", ArgSpec::Between(0, 1)),
];

/// List of the above for diagnostics
const VALID_METAVAR_EXPR_NAMES: &str = "`count`, `ignore`, `index`, `len`, and `concat`";

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
        let ident = parse_ident(
            &mut iter,
            psess,
            outer_span,
            MveExpectedIdentContext::ExprName { valid_expr_list: VALID_METAVAR_EXPR_NAMES },
        )?;

        let next = iter.next();
        let Some(TokenTree::Delimited(.., Delimiter::Parenthesis, args)) = next else {
            // No `()`; wrong or no delimiters. Point at a problematic span or a place to
            // add parens if it makes sense.
            let (unexpected_span, insert_span) = match next {
                Some(TokenTree::Delimited(..)) => (None, None),
                Some(tt) => (Some(tt.span()), None),
                None => (None, Some(ident.span.shrink_to_hi())),
            };
            let err =
                errors::MveMissingParen { ident_span: ident.span, unexpected_span, insert_span };
            return Err(psess.dcx().create_err(err));
        };

        // Ensure there are no trailing tokens in the braces, e.g. `${foo() extra}`
        if iter.peek().is_some() {
            let span = iter_span(&iter).expect("checked is_some above");
            let err = errors::MveExtraTokens {
                span,
                ident_span: ident.span,
                extra_count: iter.count(),
                ..Default::default()
            };
            return Err(psess.dcx().create_err(err));
        }

        let mut iter = args.iter();
        let rslt = match ident.as_str() {
            "concat" => parse_concat(&mut iter, psess, outer_span, ident.span)?,
            "count" => parse_count(&mut iter, psess, ident.span)?,
            "ignore" => {
                eat_dollar(&mut iter, psess, ident.span)?;
                let ident =
                    parse_ident(&mut iter, psess, outer_span, MveExpectedIdentContext::Ignore)?;
                MetaVarExpr::Ignore(ident)
            }
            "index" => MetaVarExpr::Index(parse_depth(&mut iter, psess, ident.span)?),
            "len" => MetaVarExpr::Len(parse_depth(&mut iter, psess, ident.span)?),
            _ => {
                let err = errors::MveUnrecognizedExpr {
                    span: ident.span,
                    valid_expr_list: VALID_METAVAR_EXPR_NAMES,
                };
                return Err(psess.dcx().create_err(err));
            }
        };
        check_trailing_tokens(&mut iter, psess, ident)?;
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

/// Checks if there are any remaining tokens (for example, `${ignore($valid, extra)}`) and create
/// a diag with the correct arg count if so.
fn check_trailing_tokens<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    ident: Ident,
) -> PResult<'psess, ()> {
    if iter.peek().is_none() {
        // All tokens consumed, as expected
        return Ok(());
    }

    let (name, spec) = EXPR_NAME_ARG_MAP
        .iter()
        .find(|(name, _)| *name == ident.as_str())
        .expect("called with an invalid name");

    let (min_or_exact_args, max_args) = match *spec {
        // For expressions like `concat`, all tokens should be consumed already
        ArgSpec::Any => panic!("{name} takes unlimited tokens but didn't eat them all"),
        ArgSpec::Between(min, max) => (min, Some(max)),
        ArgSpec::Exact(n) => (n, None),
    };

    let err = errors::MveExtraTokens {
        span: iter_span(iter).expect("checked is_none above"),
        ident_span: ident.span,
        extra_count: iter.count(),

        exact_args_note: if max_args.is_some() { None } else { Some(()) },
        range_args_note: if max_args.is_some() { Some(()) } else { None },
        min_or_exact_args,
        max_args: max_args.unwrap_or_default(),
        name,
    };
    Err(psess.dcx().create_err(err))
}

/// Returns a span encompassing all tokens in the iterator if there is at least one item.
fn iter_span(iter: &TokenStreamIter<'_>) -> Option<Span> {
    let mut iter = iter.clone(); // cloning is cheap
    let first_sp = iter.next()?.span();
    let last_sp = iter.last().map(TokenTree::span).unwrap_or(first_sp);
    let span = first_sp.with_hi(last_sp.hi());
    Some(span)
}

/// Indicates what is placed in a `concat` parameter. For example, literals
/// (`${concat("foo", "bar")}`) or adhoc identifiers (`${concat(foo, bar)}`).
#[derive(Debug, Decodable, Encodable, PartialEq)]
pub(crate) enum MetaVarExprConcatElem {
    /// Identifier WITHOUT a preceding dollar sign, which means that this identifier should be
    /// interpreted as a literal.
    Ident(String),
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
    let dcx = psess.dcx();
    loop {
        let dollar = try_eat_dollar(iter);
        let Some(tt) = iter.next() else {
            // May be hit only with the first iteration (peek is otherwise checked at the end).
            break;
        };

        let make_err = |reason| {
            let err = errors::MveConcatInvalid {
                span: tt.span(),
                ident_span: expr_ident_span,
                reason,
                valid: VALID_EXPR_CONCAT_TYPES,
            };
            Err(dcx.create_err(err))
        };

        let token = match tt {
            TokenTree::Token(token, _) => token,
            TokenTree::Delimited(..) => {
                return make_err(MveConcatInvalidReason::UnexpectedGroup);
            }
        };

        let element = if let Some(dollar) = dollar {
            // Expecting a metavar
            let Some((ident, _)) = token.ident() else {
                return make_err(MveConcatInvalidReason::ExpectedMetavarIdent {
                    found: pprust::token_to_string(token).into_owned(),
                    dollar,
                });
            };

            // Variables get passed untouched
            MetaVarExprConcatElem::Var(ident)
        } else if let TokenKind::Literal(lit) = token.kind {
            // Preprocess with `from_token_lit` to handle unescaping, float / int literal suffix
            // stripping.
            //
            // For consistent user experience, please keep this in sync with the handling of
            // literals in `rustc_builtin_macros::concat`!
            let s = match ast::LitKind::from_token_lit(lit.clone()) {
                Ok(ast::LitKind::Str(s, _)) => s.to_string(),
                Ok(ast::LitKind::Float(..)) => {
                    return make_err(MveConcatInvalidReason::FloatLit);
                }
                Ok(ast::LitKind::Char(c)) => c.to_string(),
                Ok(ast::LitKind::Int(i, _)) => i.to_string(),
                Ok(ast::LitKind::Bool(b)) => b.to_string(),
                Ok(ast::LitKind::CStr(..)) => return make_err(MveConcatInvalidReason::CStrLit),
                Ok(ast::LitKind::Byte(..) | ast::LitKind::ByteStr(..)) => {
                    return make_err(MveConcatInvalidReason::ByteStrLit);
                }
                Ok(ast::LitKind::Err(_guarantee)) => {
                    // REVIEW: a diagnostic was already emitted, should we just break?
                    return make_err(MveConcatInvalidReason::InvalidLiteral);
                }
                Err(err) => return Err(create_lit_error(psess, err, lit, token.span)),
            };

            if !s.chars().all(|ch| is_id_continue(ch)) {
                // Check that all characters are valid in the middle of an identifier. This doesn't
                // guarantee that the final identifier is valid (we still need to check it later),
                // but it allows us to catch errors with specific arguments before expansion time;
                // for example, string literal "foo.bar" gets flagged before the macro is invoked.
                return make_err(MveConcatInvalidReason::InvalidIdent);
            }

            MetaVarExprConcatElem::Ident(s)
        } else if let Some((elem, is_raw)) = token.ident() {
            if is_raw == IdentIsRaw::Yes {
                return make_err(MveConcatInvalidReason::RawIdentifier);
            }
            MetaVarExprConcatElem::Ident(elem.as_str().to_string())
        } else {
            return make_err(MveConcatInvalidReason::UnsupportedInput);
        };

        result.push(element);

        if iter.peek().is_none() {
            // break before trying to eat the comma
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
    let ident = parse_ident(iter, psess, span, MveExpectedIdentContext::Count)?;
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

/// Tries to parse a generic ident. If this fails, create a missing identifier diagnostic with
/// `context` explanation.
fn parse_ident<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    fallback_span: Span,
    context: MveExpectedIdentContext,
) -> PResult<'psess, Ident> {
    let Some(tt) = iter.next() else {
        let err = errors::MveExpectedIdent { span: fallback_span, not_ident_label: None, context };
        return Err(psess.dcx().create_err(err));
    };

    let TokenTree::Token(token, _) = tt else {
        let span = tt.span();
        let err = errors::MveExpectedIdent { span, not_ident_label: Some(span), context };
        return Err(psess.dcx().create_err(err));
    };

    let Some((elem, _)) = token.ident() else {
        let span = token.span;
        let err = errors::MveExpectedIdent { span, not_ident_label: Some(span), context };
        return Err(psess.dcx().create_err(err));
    };

    Ok(elem)
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

/// Tries to move the iterator forward returning `Some(dollar_span)` if there is a dollar sign. If
/// not, then the iterator is not modified and the result is `None`.
fn try_eat_dollar(iter: &mut TokenStreamIter<'_>) -> Option<Span> {
    if let Some(TokenTree::Token(Token { kind: token::Dollar, span }, _)) = iter.peek() {
        let _ = iter.next();
        return Some(*span);
    }
    None
}

/// Expects that the next item is a dollar sign.
fn eat_dollar<'psess>(
    iter: &mut TokenStreamIter<'_>,
    psess: &'psess ParseSess,
    span: Span,
) -> PResult<'psess, ()> {
    if try_eat_dollar(iter).is_some() {
        return Ok(());
    }
    Err(psess.dcx().struct_span_err(
        span,
        "meta-variables within meta-variable expressions must be referenced using a dollar sign",
    ))
}
