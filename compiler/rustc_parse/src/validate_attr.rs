//! Meta-syntax validation logic of attributes for post-expansion.

use crate::{errors, parse_in};

use rustc_ast::tokenstream::DelimSpan;
use rustc_ast::MetaItemKind;
use rustc_ast::{self as ast, AttrArgs, AttrArgsEq, Attribute, DelimArgs, MacDelimiter, MetaItem};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, FatalError, PResult};
use rustc_feature::{AttributeTemplate, BuiltinAttribute, BUILTIN_ATTRIBUTE_MAP};
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;
use rustc_session::parse::ParseSess;
use rustc_span::{sym, Span, Symbol};

pub fn check_attr(sess: &ParseSess, attr: &Attribute) {
    if attr.is_doc_comment() {
        return;
    }

    let attr_info = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

    // Check input tokens for built-in and key-value attributes.
    match attr_info {
        // `rustc_dummy` doesn't have any restrictions specific to built-in attributes.
        Some(BuiltinAttribute { name, template, .. }) if *name != sym::rustc_dummy => {
            check_builtin_attribute(sess, attr, *name, *template)
        }
        _ if let AttrArgs::Eq(..) = attr.get_normal_item().args => {
            // All key-value attributes are restricted to meta-item syntax.
            parse_meta(sess, attr)
                .map_err(|mut err| {
                    err.emit();
                })
                .ok();
        }
        _ => {}
    }
}

pub fn parse_meta<'a>(sess: &'a ParseSess, attr: &Attribute) -> PResult<'a, MetaItem> {
    let item = attr.get_normal_item();
    Ok(MetaItem {
        span: attr.span,
        path: item.path.clone(),
        kind: match &item.args {
            AttrArgs::Empty => MetaItemKind::Word,
            AttrArgs::Delimited(DelimArgs { dspan, delim, tokens }) => {
                check_meta_bad_delim(sess, *dspan, *delim);
                let nmis = parse_in(sess, tokens.clone(), "meta list", |p| p.parse_meta_seq_top())?;
                MetaItemKind::List(nmis)
            }
            AttrArgs::Eq(_, AttrArgsEq::Ast(expr)) => {
                if let ast::ExprKind::Lit(token_lit) = expr.kind
                    && let Ok(lit) = ast::MetaItemLit::from_token_lit(token_lit, expr.span)
                {
                    if token_lit.suffix.is_some() {
                        let mut err = sess.span_diagnostic.struct_span_err(
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
                } else {
                    // The non-error case can happen with e.g. `#[foo = 1+1]`. The error case can
                    // happen with e.g. `#[foo = include_str!("nonexistent-file.rs")]`; in that
                    // case we delay the error because an earlier error will have already been
                    // reported.
                    let msg = format!("unexpected expression: `{}`", pprust::expr_to_string(expr));
                    let mut err = sess.span_diagnostic.struct_span_err(expr.span, msg);
                    if let ast::ExprKind::Err = expr.kind {
                        err.downgrade_to_delayed_bug();
                    }
                    return Err(err);
                }
            }
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => MetaItemKind::NameValue(lit.clone()),
        },
    })
}

pub fn check_meta_bad_delim(sess: &ParseSess, span: DelimSpan, delim: MacDelimiter) {
    if let ast::MacDelimiter::Parenthesis = delim {
        return;
    }
    sess.emit_err(errors::MetaBadDelim {
        span: span.entire(),
        sugg: errors::MetaBadDelimSugg { open: span.open, close: span.close },
    });
}

pub fn check_cfg_attr_bad_delim(sess: &ParseSess, span: DelimSpan, delim: MacDelimiter) {
    if let ast::MacDelimiter::Parenthesis = delim {
        return;
    }
    sess.emit_err(errors::CfgAttrBadDelim {
        span: span.entire(),
        sugg: errors::MetaBadDelimSugg { open: span.open, close: span.close },
    });
}

/// Checks that the given meta-item is compatible with this `AttributeTemplate`.
fn is_attr_template_compatible(template: &AttributeTemplate, meta: &ast::MetaItemKind) -> bool {
    match meta {
        MetaItemKind::Word => template.word,
        MetaItemKind::List(..) => template.list.is_some(),
        MetaItemKind::NameValue(lit) if lit.kind.is_str() => template.name_value_str.is_some(),
        MetaItemKind::NameValue(..) => false,
    }
}

pub fn check_builtin_attribute(
    sess: &ParseSess,
    attr: &Attribute,
    name: Symbol,
    template: AttributeTemplate,
) {
    match parse_meta(sess, attr) {
        Ok(meta) => check_builtin_meta_item(sess, &meta, attr.style, name, template),
        Err(mut err) => {
            err.emit();
        }
    }
}

pub fn check_builtin_meta_item(
    sess: &ParseSess,
    meta: &MetaItem,
    style: ast::AttrStyle,
    name: Symbol,
    template: AttributeTemplate,
) {
    // Some special attributes like `cfg` must be checked
    // before the generic check, so we skip them here.
    let should_skip = |name| name == sym::cfg;

    if !should_skip(name) && !is_attr_template_compatible(&template, &meta.kind) {
        emit_malformed_attribute(sess, style, meta.span, name, template);
    }
}

fn emit_malformed_attribute(
    sess: &ParseSess,
    style: ast::AttrStyle,
    span: Span,
    name: Symbol,
    template: AttributeTemplate,
) {
    // Some of previously accepted forms were used in practice,
    // report them as warnings for now.
    let should_warn = |name| {
        matches!(name, sym::doc | sym::ignore | sym::inline | sym::link | sym::test | sym::bench)
    };

    let error_msg = format!("malformed `{name}` attribute input");
    let mut msg = "attribute must be of the form ".to_owned();
    let mut suggestions = vec![];
    let mut first = true;
    let inner = if style == ast::AttrStyle::Inner { "!" } else { "" };
    if template.word {
        first = false;
        let code = format!("#{inner}[{name}]");
        msg.push_str(&format!("`{code}`"));
        suggestions.push(code);
    }
    if let Some(descr) = template.list {
        if !first {
            msg.push_str(" or ");
        }
        first = false;
        let code = format!("#{inner}[{name}({descr})]");
        msg.push_str(&format!("`{code}`"));
        suggestions.push(code);
    }
    if let Some(descr) = template.name_value_str {
        if !first {
            msg.push_str(" or ");
        }
        let code = format!("#{inner}[{name} = \"{descr}\"]");
        msg.push_str(&format!("`{code}`"));
        suggestions.push(code);
    }
    if should_warn(name) {
        sess.buffer_lint(&ILL_FORMED_ATTRIBUTE_INPUT, span, ast::CRATE_NODE_ID, msg);
    } else {
        sess.span_diagnostic
            .struct_span_err(span, error_msg)
            .span_suggestions(
                span,
                if suggestions.len() == 1 {
                    "must be of the form"
                } else {
                    "the following are the possible correct uses"
                },
                suggestions.into_iter(),
                Applicability::HasPlaceholders,
            )
            .emit();
    }
}

pub fn emit_fatal_malformed_builtin_attribute(
    sess: &ParseSess,
    attr: &Attribute,
    name: Symbol,
) -> ! {
    let template = BUILTIN_ATTRIBUTE_MAP.get(&name).expect("builtin attr defined").template;
    emit_malformed_attribute(sess, attr.style, attr.span, name, template);
    // This is fatal, otherwise it will likely cause a cascade of other errors
    // (and an error here is expected to be very rare).
    FatalError.raise()
}
