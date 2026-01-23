#![allow(warnings)]
use rustc_hir::attrs::diagnostic::OnUnimplementedDirective;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::{
    MALFORMED_DIAGNOSTIC_ATTRIBUTES, MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
};
use thin_vec::thin_vec;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::attributes::template;
/// Folds all uses of `#[rustc_on_unimplemented]` and `#[diagnostic::on_unimplemented]`.
/// TODO: example
#[derive(Default)]
pub struct OnUnimplementedParser {
    directive: Option<(Span, OnUnimplementedDirective)>,
}

impl<S: Stage> AttributeParser<S> for OnUnimplementedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (&[sym::diagnostic, sym::on_unimplemented], template!(Word), |this, cx, args| {
            let span = cx.attr_span;

            let items = match args {
                ArgParser::List(items) => items,
                ArgParser::NoArgs => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MissingOptionsForOnUnimplemented,
                        span,
                    );
                    return;
                }
                ArgParser::NameValue(_) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnUnimplementedAttr { span },
                        span,
                    );
                    return;
                }
            };

            let Some(directive) = parse_directive_items(cx, Ctx::DiagnosticOnUnimplemented, items)
            else {
                return;
            };
            merge_directives(cx, &mut this.directive, (span, directive));
        }),
        // todo (&[sym::rustc_on_unimplemented], template!(Word), |this, cx, args| {}),
    ];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(mut self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        self.directive.map(|(span, directive)| AttributeKind::OnUnimplemented {
            span,
            directive: Some(Box::new(directive)),
        })
    }
}

fn merge_directives<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    first: &mut Option<(Span, OnUnimplementedDirective)>,
    later: (Span, OnUnimplementedDirective),
) {
    if let Some((first_span, first)) = first {
        merge(cx, &mut first.message, later.1.message, "message");
        merge(cx, &mut first.label, later.1.label, "label");
        first.notes.extend(later.1.notes);
    } else {
        *first = Some(later);
    }
}

fn merge<T, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    first: &mut Option<(Span, T)>,
    later: Option<(Span, T)>,
    option_name: &'static str,
) {
    match (first, later) {
        (Some(_) | None, None) => {}
        (Some((first_span, _)), Some((later_span, _))) => {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::IgnoredDiagnosticOption {
                    first_span: *first_span,
                    later_span,
                    option_name,
                },
                later_span,
            );
        }
        (first @ None, Some(later)) => {
            first.get_or_insert(later);
        }
    }
}

fn parse_directive_items<S: Stage>(
    cx: &mut AcceptContext<S>,
    ctx: Ctx,
    items: &MetaItemListParser,
) -> Option<OnUnimplementedDirective> {
    let condition = None;
    let mut message = None;
    let mut label = None;
    let mut notes = ThinVec::new();
    let mut parent_label = None;
    let mut subcommands = ThinVec::new();
    let mut append_const_msg = None;

    for item in items.mixed() {
        // At this point, we are expecting any of:
        // message = "..", label = "..", note = ".."
        let Some((name, value, value_span)) = (try {
            let item = item.meta_item()?;
            let name = item.ident()?.name;
            let nv = item.args().name_value()?;
            let value = nv.value_as_str()?;
            (name, value, nv.value_span)
        }) else {
            let span = item.span();
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::MalformedOnUnimplementedAttr { span },
                span,
            );
            continue;
        };

        let mut parse = |input| {
            let snippet = cx.sess.source_map().span_to_snippet(value_span).ok();
            match parse_format_string(input, snippet, value_span, ctx) {
                Ok((f, warnings)) => {
                    for warning in warnings {
                        let (FormatWarning::InvalidSpecifier { span, .. }
                        | FormatWarning::PositionalArgument { span, .. }) = warning;
                        cx.emit_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            AttributeLintKind::MalformedDiagnosticFormat { warning },
                            span,
                        );
                    }

                    f
                }
                Err(e) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                        AttributeLintKind::DiagnosticWrappedParserError {
                            description: e.description,
                            label: e.label,
                        },
                        value_span,
                    );
                    // We could not parse the input, just use it as-is.
                    FormatString { input, span: value_span, pieces: thin_vec![Piece::Lit(input)] }
                }
            }
        };
        match name {
            sym::message => {
                if message.is_none() {
                    message.insert((item.span(), parse(value)));
                } else {
                    // warn
                }
            }
            sym::label => {
                if label.is_none() {
                    label.insert((item.span(), parse(value)));
                } else {
                    // warn
                }
            }
            sym::note => notes.push(parse(value)),
            _other => {
                let span = item.span();
                cx.emit_lint(
                    MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    AttributeLintKind::MalformedOnUnimplementedAttr { span },
                    span,
                );
                continue;
            }
        }
    }

    Some(OnUnimplementedDirective {
        condition,
        subcommands,
        message,
        label,
        notes,
        parent_label,
        append_const_msg,
    })
}
