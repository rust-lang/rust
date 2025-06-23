//! Meta-syntax validation logic of attributes for post-expansion.

use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::DelimSpan;
use rustc_ast::{
    self as ast, AttrArgs, Attribute, DelimArgs, MetaItem, MetaItemInner, MetaItemKind, NodeId,
    Safety,
};
use rustc_errors::{Applicability, FatalError, PResult};
use rustc_feature::{AttributeSafety, AttributeTemplate, BUILTIN_ATTRIBUTE_MAP, BuiltinAttribute};
use rustc_session::errors::report_lit_error;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::{ILL_FORMED_ATTRIBUTE_INPUT, UNSAFE_ATTR_OUTSIDE_UNSAFE};
use rustc_session::parse::ParseSess;
use rustc_span::{Span, Symbol, sym};

use crate::{errors, parse_in};

pub fn check_attr(psess: &ParseSess, attr: &Attribute, id: NodeId) {
    if attr.is_doc_comment() || attr.has_name(sym::cfg_trace) || attr.has_name(sym::cfg_attr_trace)
    {
        return;
    }

    let builtin_attr_info = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

    let builtin_attr_safety = builtin_attr_info.map(|x| x.safety);
    check_attribute_safety(psess, builtin_attr_safety, attr, id);

    // Check input tokens for built-in and key-value attributes.
    match builtin_attr_info {
        // `rustc_dummy` doesn't have any restrictions specific to built-in attributes.
        Some(BuiltinAttribute { name, template, .. }) if *name != sym::rustc_dummy => {
            match parse_meta(psess, attr) {
                // Don't check safety again, we just did that
                Ok(meta) => {
                    check_builtin_meta_item(psess, &meta, attr.style, *name, *template, false)
                }
                Err(err) => {
                    err.emit();
                }
            }
        }
        _ => {
            let attr_item = attr.get_normal_item();
            if let AttrArgs::Eq { .. } = attr_item.args {
                // All key-value attributes are restricted to meta-item syntax.
                match parse_meta(psess, attr) {
                    Ok(_) => {}
                    Err(err) => {
                        err.emit();
                    }
                }
            }
        }
    }
}

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

/// Checks that the given meta-item is compatible with this `AttributeTemplate`.
fn is_attr_template_compatible(template: &AttributeTemplate, meta: &ast::MetaItemKind) -> bool {
    let is_one_allowed_subword = |items: &[MetaItemInner]| match items {
        [item] => item.is_word() && template.one_of.iter().any(|&word| item.has_name(word)),
        _ => false,
    };
    match meta {
        MetaItemKind::Word => template.word,
        MetaItemKind::List(items) => template.list.is_some() || is_one_allowed_subword(items),
        MetaItemKind::NameValue(lit) if lit.kind.is_str() => template.name_value_str.is_some(),
        MetaItemKind::NameValue(..) => false,
    }
}

pub fn check_attribute_safety(
    psess: &ParseSess,
    builtin_attr_safety: Option<AttributeSafety>,
    attr: &Attribute,
    id: NodeId,
) {
    let attr_item = attr.get_normal_item();
    match (builtin_attr_safety, attr_item.unsafety) {
        // - Unsafe builtin attribute
        // - User wrote `#[unsafe(..)]`, which is permitted on any edition
        (Some(AttributeSafety::Unsafe { .. }), Safety::Unsafe(..)) => {
            // OK
        }

        // - Unsafe builtin attribute
        // - User did not write `#[unsafe(..)]`
        (Some(AttributeSafety::Unsafe { unsafe_since }), Safety::Default) => {
            let path_span = attr_item.path.span;

            // If the `attr_item`'s span is not from a macro, then just suggest
            // wrapping it in `unsafe(...)`. Otherwise, we suggest putting the
            // `unsafe(`, `)` right after and right before the opening and closing
            // square bracket respectively.
            let diag_span = attr_item.span();

            // Attributes can be safe in earlier editions, and become unsafe in later ones.
            //
            // Use the span of the attribute's name to determine the edition: the span of the
            // attribute as a whole may be inaccurate if it was emitted by a macro.
            //
            // See https://github.com/rust-lang/rust/issues/142182.
            let emit_error = match unsafe_since {
                None => true,
                Some(unsafe_since) => path_span.edition() >= unsafe_since,
            };

            if emit_error {
                psess.dcx().emit_err(errors::UnsafeAttrOutsideUnsafe {
                    span: path_span,
                    suggestion: errors::UnsafeAttrOutsideUnsafeSuggestion {
                        left: diag_span.shrink_to_lo(),
                        right: diag_span.shrink_to_hi(),
                    },
                });
            } else {
                psess.buffer_lint(
                    UNSAFE_ATTR_OUTSIDE_UNSAFE,
                    path_span,
                    id,
                    BuiltinLintDiag::UnsafeAttrOutsideUnsafe {
                        attribute_name_span: path_span,
                        sugg_spans: (diag_span.shrink_to_lo(), diag_span.shrink_to_hi()),
                    },
                );
            }
        }

        // - Normal builtin attribute, or any non-builtin attribute
        // - All non-builtin attributes are currently considered safe; writing `#[unsafe(..)]` is
        //   not permitted on non-builtin attributes or normal builtin attributes
        (Some(AttributeSafety::Normal) | None, Safety::Unsafe(unsafe_span)) => {
            psess.dcx().emit_err(errors::InvalidAttrUnsafe {
                span: unsafe_span,
                name: attr_item.path.clone(),
            });
        }

        // - Normal builtin attribute
        // - No explicit `#[unsafe(..)]` written.
        (Some(AttributeSafety::Normal), Safety::Default) => {
            // OK
        }

        // - Non-builtin attribute
        // - No explicit `#[unsafe(..)]` written.
        (None, Safety::Default) => {
            // OK
        }

        (
            Some(AttributeSafety::Unsafe { .. } | AttributeSafety::Normal) | None,
            Safety::Safe(..),
        ) => {
            psess.dcx().span_delayed_bug(
                attr_item.span(),
                "`check_attribute_safety` does not expect `Safety::Safe` on attributes",
            );
        }
    }
}

// Called by `check_builtin_meta_item` and code that manually denies
// `unsafe(...)` in `cfg`
pub fn deny_builtin_meta_unsafety(psess: &ParseSess, meta: &MetaItem) {
    // This only supports denying unsafety right now - making builtin attributes
    // support unsafety will requite us to thread the actual `Attribute` through
    // for the nice diagnostics.
    if let Safety::Unsafe(unsafe_span) = meta.unsafety {
        psess
            .dcx()
            .emit_err(errors::InvalidAttrUnsafe { span: unsafe_span, name: meta.path.clone() });
    }
}

pub fn check_builtin_meta_item(
    psess: &ParseSess,
    meta: &MetaItem,
    style: ast::AttrStyle,
    name: Symbol,
    template: AttributeTemplate,
    deny_unsafety: bool,
) {
    if !is_attr_template_compatible(&template, &meta.kind) {
        emit_malformed_attribute(psess, style, meta.span, name, template);
    }

    if deny_unsafety {
        deny_builtin_meta_unsafety(psess, meta);
    }
}

fn emit_malformed_attribute(
    psess: &ParseSess,
    style: ast::AttrStyle,
    span: Span,
    name: Symbol,
    template: AttributeTemplate,
) {
    // attrs with new parsers are locally validated so excluded here
    if matches!(
        name,
        sym::inline
            | sym::rustc_force_inline
            | sym::rustc_confusables
            | sym::repr
            | sym::align
            | sym::deprecated
            | sym::optimize
            | sym::cold
            | sym::must_use
    ) {
        return;
    }

    // Some of previously accepted forms were used in practice,
    // report them as warnings for now.
    let should_warn =
        |name| matches!(name, sym::doc | sym::ignore | sym::link | sym::test | sym::bench);

    let error_msg = format!("malformed `{name}` attribute input");
    let mut suggestions = vec![];
    let inner = if style == ast::AttrStyle::Inner { "!" } else { "" };
    if template.word {
        suggestions.push(format!("#{inner}[{name}]"));
    }
    if let Some(descr) = template.list {
        suggestions.push(format!("#{inner}[{name}({descr})]"));
    }
    suggestions.extend(template.one_of.iter().map(|&word| format!("#{inner}[{name}({word})]")));
    if let Some(descr) = template.name_value_str {
        suggestions.push(format!("#{inner}[{name} = \"{descr}\"]"));
    }
    if should_warn(name) {
        psess.buffer_lint(
            ILL_FORMED_ATTRIBUTE_INPUT,
            span,
            ast::CRATE_NODE_ID,
            BuiltinLintDiag::IllFormedAttributeInput { suggestions: suggestions.clone() },
        );
    } else {
        suggestions.sort();
        psess
            .dcx()
            .struct_span_err(span, error_msg)
            .with_span_suggestions(
                span,
                if suggestions.len() == 1 {
                    "must be of the form"
                } else {
                    "the following are the possible correct uses"
                },
                suggestions,
                Applicability::HasPlaceholders,
            )
            .emit();
    }
}

pub fn emit_fatal_malformed_builtin_attribute(
    psess: &ParseSess,
    attr: &Attribute,
    name: Symbol,
) -> ! {
    let template = BUILTIN_ATTRIBUTE_MAP.get(&name).expect("builtin attr defined").template;
    emit_malformed_attribute(psess, attr.style, attr.span, name, template);
    // This is fatal, otherwise it will likely cause a cascade of other errors
    // (and an error here is expected to be very rare).
    FatalError.raise()
}
