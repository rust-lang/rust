//! Meta-syntax validation logic of attributes for post-expansion.

use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::DelimSpan;
use rustc_ast::{self as ast, AttrArgs, Attribute, DelimArgs, MetaItem, MetaItemKind};
use rustc_errors::PResult;
use rustc_session::errors::report_lit_error;
use rustc_session::parse::ParseSess;

use crate::{errors, parse_in};

pub fn parse_meta<'a>(psess: &'a ParseSess, attr: &Attribute) -> PResult<'a, MetaItem> {
    let item = attr.get_normal_item();
    Ok(MetaItem {
        unsafety: item.unsafety,
        span: attr.span,
        path: item.path.clone(),
        kind: match &item.args {
            AttrArgs::Empty => MetaItemKind::Word,
            AttrArgs::Delimited(DelimArgs { dspan, delim, tokens }) => {
                check_meta_bad_delim(psess, *dspan, *delim);
                let nmis =
                    parse_in(psess, tokens.clone(), "meta list", |p| p.parse_meta_seq_top())?;
                MetaItemKind::List(nmis)
            }
            AttrArgs::Eq { expr, .. } => {
                if let ast::ExprKind::Lit(token_lit) = expr.kind {
                    let res = ast::MetaItemLit::from_token_lit(token_lit, expr.span);
                    let res = match res {
                        Ok(lit) => {
                            if token_lit.suffix.is_some() {
                                let mut err = psess.dcx().struct_span_err(
                                    expr.span,
                                    "suffixed literals are not allowed in attributes",
                                );
                                err.help(
                                    "instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), \
                                    use an unsuffixed version (`1`, `1.0`, etc.)",
                                );
                                return Err(err);
                            } else {
                                MetaItemKind::NameValue(lit)
                            }
                        }
                        Err(err) => {
                            let guar = report_lit_error(psess, err, token_lit, expr.span);
                            let lit = ast::MetaItemLit {
                                symbol: token_lit.symbol,
                                suffix: token_lit.suffix,
                                kind: ast::LitKind::Err(guar),
                                span: expr.span,
                            };
                            MetaItemKind::NameValue(lit)
                        }
                    };
                    res
                } else {
                    // Example cases:
                    // - `#[foo = 1+1]`: results in `ast::ExprKind::BinOp`.
                    // - `#[foo = include_str!("nonexistent-file.rs")]`:
                    //   results in `ast::ExprKind::Err`. In that case we delay
                    //   the error because an earlier error will have already
                    //   been reported.
                    let msg = "attribute value must be a literal";
                    let mut err = psess.dcx().struct_span_err(expr.span, msg);
                    if let ast::ExprKind::Err(_) = expr.kind {
                        err.downgrade_to_delayed_bug();
                    }
                    return Err(err);
                }
            }
        },
    })
}

fn check_meta_bad_delim(psess: &ParseSess, span: DelimSpan, delim: Delimiter) {
    if let Delimiter::Parenthesis = delim {
        return;
    }
    psess.dcx().emit_err(errors::MetaBadDelim {
        span: span.entire(),
        sugg: errors::MetaBadDelimSugg { open: span.open, close: span.close },
    });
}

pub(super) fn check_cfg_attr_bad_delim(psess: &ParseSess, span: DelimSpan, delim: Delimiter) {
    if let Delimiter::Parenthesis = delim {
        return;
    }
    psess.dcx().emit_err(errors::CfgAttrBadDelim {
        span: span.entire(),
        sugg: errors::MetaBadDelimSugg { open: span.open, close: span.close },
    });
}
