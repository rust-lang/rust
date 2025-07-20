use rustc_ast::{self as ast, MetaItem, MetaItemInner, MetaItemKind, Safety};
use rustc_attr_data_structures::lints::{AttributeLint, AttributeLintKind};
use rustc_errors::Applicability;
use rustc_feature::{AttributeSafety, AttributeTemplate, BUILTIN_ATTRIBUTE_MAP, BuiltinAttribute};
use rustc_parse::validate_attr::parse_meta;
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;
use rustc_span::fatal_error::FatalError;
use rustc_span::{Span, Symbol, sym};

use crate::context::Stage;
use crate::session_diagnostics::{
    InvalidAttrUnsafe, UnsafeAttrOutsideUnsafe, UnsafeAttrOutsideUnsafeSuggestion,
};
use crate::{AttributeParser, Early};

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub(crate) fn validate_attribute(
        &self,
        attr: &ast::Attribute,
        target_id: S::Id,
        emit_lint: &mut impl FnMut(AttributeLint<S::Id>),
        skip_template_check: bool,
    ) -> bool {
        if !self.stage().should_emit_errors_and_lints() {
            return true;
        }
        let ast::AttrKind::Normal(normal) = &attr.kind else { return true };

        // Trace attributes have already been checked when their respective non-trace attribute is parsed
        if attr.has_name(sym::cfg_trace) || attr.has_name(sym::cfg_attr_trace) {
            return true;
        }

        // Check attribute for safety
        check_attribute_safety::<S>(self.sess(), attr, target_id, emit_lint);
        let builtin_attr_info =
            attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

        // All non-builtin attributes (and rustc_dummy) are not checked further unless they're for the form `#[attr = ...]`.
        if (builtin_attr_info.is_none() || attr.has_name(sym::rustc_dummy))
            && !matches!(normal.item.args, rustc_ast::AttrArgs::Eq { .. })
        {
            return true;
        }

        // FIXME This check will be removed in the future (and costs performance :c)
        // It checks some general things about the attribute (such that the literals are valid)
        // This will be checked by the attribute parser in the future
        let meta = match parse_meta(&self.sess().psess, attr) {
            Ok(meta) => meta,
            Err(err) => {
                err.emit();
                return false;
            }
        };

        // FIXME when all attributes are parsed, this check can be removed
        // this checks the structure of built-in attributes that are not parsed yet
        if skip_template_check {
            return true;
        }
        if let Some(BuiltinAttribute { name, template, .. }) = builtin_attr_info {
            if !is_attr_template_compatible(&template, &meta.kind) {
                emit_malformed_attribute::<S>(
                    self.sess(),
                    attr.style,
                    meta.span,
                    *name,
                    *template,
                    &mut |suggestions| {
                        emit_lint(AttributeLint {
                            id: target_id,
                            span: meta.span,
                            kind: AttributeLintKind::IllFormedAttributeInput { suggestions },
                        })
                    },
                );
                return false;
            }
        }

        true
    }
}

// FIXME this function can be removed when all attributes are parsed
// For now it is still used by a few built-in unparsed attributes
/// Check the safety and template of a given MetaItem
pub fn check_builtin_meta_item(
    sess: &Session,
    meta: &MetaItem,
    style: ast::AttrStyle,
    name: Symbol,
    template: AttributeTemplate,
) {
    if !is_attr_template_compatible(&template, &meta.kind) {
        emit_malformed_attribute::<Early>(
            sess,
            style,
            meta.span,
            name,
            template,
            &mut |suggestions| {
                sess.psess.buffer_lint(
                    ILL_FORMED_ATTRIBUTE_INPUT,
                    meta.span,
                    ast::CRATE_NODE_ID,
                    BuiltinLintDiag::IllFormedAttributeInput { suggestions },
                )
            },
        );
    }
    // This only supports denying unsafety right now - making builtin attributes
    // support unsafety will requite us to thread the actual `Attribute` through
    // for the nice diagnostics.
    if let Safety::Unsafe(unsafe_span) = meta.unsafety {
        sess.dcx().emit_err(InvalidAttrUnsafe { span: unsafe_span, name: meta.path.clone() });
    }
}

// FIXME this function can be removed when all attributes are parsed
// For now it is still used by a few built-in unparsed attributes
pub fn emit_fatal_malformed_builtin_attribute(
    sess: &Session,
    attr: &rustc_ast::Attribute,
    name: Symbol,
) -> ! {
    let template = BUILTIN_ATTRIBUTE_MAP.get(&name).expect("builtin attr defined").template;
    emit_malformed_attribute::<Early>(sess, attr.style, attr.span, name, template, &mut |_| {});
    // This is fatal, otherwise it will likely cause a cascade of other errors
    // (and an error here is expected to be very rare).
    FatalError.raise()
}

fn emit_malformed_attribute<S: Stage>(
    sess: &Session,
    style: ast::AttrStyle,
    span: Span,
    name: Symbol,
    template: AttributeTemplate,
    emit_lint: &mut impl FnMut(Vec<String>),
) {
    // Some of previously accepted forms were used in practice,
    // report them as warnings for now.
    let should_warn = |name| matches!(name, sym::doc | sym::link | sym::test | sym::bench);

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
        emit_lint(suggestions);
    } else {
        suggestions.sort();
        sess.dcx()
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

// FIXME this function can be made private when all attributes are parsed
// For now it is still used by a few built-in unparsed attributes
pub fn check_attribute_safety<S: Stage>(
    sess: &Session,
    attr: &ast::Attribute,
    id: S::Id,
    emit_lint: &mut impl FnMut(AttributeLint<S::Id>),
) {
    let builtin_attr_info = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

    let builtin_attr_safety = builtin_attr_info.map(|x| x.safety);

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
                sess.dcx().emit_err(UnsafeAttrOutsideUnsafe {
                    span: path_span,
                    suggestion: UnsafeAttrOutsideUnsafeSuggestion {
                        left: diag_span.shrink_to_lo(),
                        right: diag_span.shrink_to_hi(),
                    },
                });
            } else {
                emit_lint(AttributeLint {
                    id,
                    span: path_span,
                    kind: AttributeLintKind::UnsafeAttrOutsideUnsafe {
                        attribute_name_span: path_span,
                        sugg_spans: (diag_span.shrink_to_lo(), diag_span.shrink_to_hi()),
                    },
                });
            }
        }

        // - Normal builtin attribute, or any non-builtin attribute
        // - All non-builtin attributes are currently considered safe; writing `#[unsafe(..)]` is
        //   not permitted on non-builtin attributes or normal builtin attributes
        (Some(AttributeSafety::Normal) | None, Safety::Unsafe(unsafe_span)) => {
            sess.dcx()
                .emit_err(InvalidAttrUnsafe { span: unsafe_span, name: attr_item.path.clone() });
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
            sess.psess.dcx().span_delayed_bug(
                attr_item.span(),
                "`check_attribute_safety` does not expect `Safety::Safe` on attributes",
            );
        }
    }
}

/// Checks that the given meta-item is compatible with this `AttributeTemplate`.
fn is_attr_template_compatible(template: &AttributeTemplate, meta: &MetaItemKind) -> bool {
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
