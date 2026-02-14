use rustc_hir::attrs::diagnostic::{AppendConstMessage, OnUnimplementedDirective};
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::{
    MALFORMED_DIAGNOSTIC_ATTRIBUTES, MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
};
use thin_vec::thin_vec;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::attributes::template;
/// Folds all uses of `#[rustc_on_unimplemented]` and `#[diagnostic::on_unimplemented]`.
/// FIXME(mejrs): add an example
#[derive(Default)]
pub struct OnUnimplementedParser {
    directive: Option<(Span, OnUnimplementedDirective)>,
}

impl OnUnimplementedParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        let span = cx.attr_span;

        let items = match args {
            ArgParser::List(items) if items.len() != 0 => items,
            ArgParser::NoArgs | ArgParser::List(_) => {
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

        let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) else {
            return;
        };
        merge_directives(cx, &mut self.directive, (span, directive));
    }
}

impl<S: Stage> AttributeParser<S> for OnUnimplementedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (&[sym::diagnostic, sym::on_unimplemented], template!(Word), |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnUnimplemented);
        }),
        (
            &[sym::rustc_on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            |this, cx, args| {
                this.parse(cx, args, Mode::RustcOnUnimplemented);
            },
        ),
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
        if first.is_rustc_attr || later.1.is_rustc_attr {
            cx.emit_err(DupesNotAllowed);
        }

        merge(cx, &mut first.message, later.1.message, sym::message);
        merge(cx, &mut first.label, later.1.label, sym::label);
        first.notes.extend(later.1.notes);
    } else {
        *first = Some(later);
    }
}

fn merge<T, S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    first: &mut Option<(Span, T)>,
    later: Option<(Span, T)>,
    option_name: Symbol,
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

fn parse_directive_items<'p, S: Stage>(
    cx: &mut AcceptContext<S>,
    mode: Mode,
    items: impl Iterator<Item = &'p MetaItemOrLitParser>,
    is_root: bool,
) -> Option<OnUnimplementedDirective> {
    let condition = None;
    let mut message: Option<(Span, _)> = None;
    let mut label: Option<(Span, _)> = None;
    let mut notes = ThinVec::new();
    let mut parent_label = None;
    let mut subcommands = ThinVec::new();
    let mut append_const_msg = None;

    for item in items {
        let span = item.span();

        macro malformed() {{
            match mode {
                Mode::RustcOnUnimplemented => {
                    cx.emit_err(NoValueInOnUnimplemented { span: item.span() });
                }
                Mode::DiagnosticOnUnimplemented => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnUnimplementedAttr { span },
                        span,
                    );
                }
            }
            continue;
        }}

        macro or_malformed($($code:tt)*) {{
            let Some(ret) = (try {
                $($code)*
            }) else {

                malformed!()
            };
            ret
        }}

        macro duplicate($name: ident, $($first_span:tt)*) {{
            match mode {
                Mode::RustcOnUnimplemented => {
                    cx.emit_err(NoValueInOnUnimplemented { span: item.span() });
                }
                Mode::DiagnosticOnUnimplemented => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::IgnoredDiagnosticOption {
                            first_span: $($first_span)*,
                            later_span: span,
                            option_name: $name,
                        },
                        span,
                    );
                }
            }
        }}

        let item: &MetaItemParser = or_malformed!(item.meta_item()?);
        let name = or_malformed!(item.ident()?).name;

        // Some things like `message = "message"` must have a value.
        // But with things like `append_const_msg` that is optional.
        let value: Option<Ident> = match item.args().name_value() {
            Some(nv) => Some(or_malformed!(nv.value_as_ident()?)),
            None => None,
        };

        let mut parse_format = |input: Ident| {
            let snippet = cx.sess.source_map().span_to_snippet(input.span).ok();
            let is_snippet = snippet.is_some();
            match parse_format_string(input.name, snippet, input.span, mode) {
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
                            span: slice_span(input.span, e.span, is_snippet),
                        },
                        input.span,
                    );
                    // We could not parse the input, just use it as-is.
                    FormatString {
                        input: input.name,
                        span: input.span,
                        pieces: thin_vec![Piece::Lit(input.name)],
                    }
                }
            }
        };
        match (mode, name) {
            (_, sym::message) => {
                let value = or_malformed!(value?);
                if let Some(message) = &message {
                    duplicate!(name, message.0)
                } else {
                    message.insert((item.span(), parse_format(value)));
                }
            }
            (_, sym::label) => {
                let value = or_malformed!(value?);
                if let Some(label) = &label {
                    duplicate!(name, label.0)
                } else {
                    label.insert((item.span(), parse_format(value)));
                }
            }
            (_, sym::note) => {
                let value = or_malformed!(value?);
                notes.push(parse_format(value))
            }

            (Mode::RustcOnUnimplemented, sym::append_const_msg) => {
                append_const_msg = if let Some(msg) = value {
                    Some(AppendConstMessage::Custom(msg.name, item.span()))
                } else {
                    Some(AppendConstMessage::Default)
                }
            }
            (Mode::RustcOnUnimplemented, sym::parent_label) => {
                let value = or_malformed!(value?);
                if parent_label.is_none() {
                    parent_label.insert(parse_format(value));
                } else {
                    // warn
                }
            }
            (Mode::RustcOnUnimplemented, sym::on) => {
                if is_root {
                    let mut items = or_malformed!(item.args().list()?);
                    let mut iter = items.mixed();
                    let condition: &MetaItemOrLitParser = match iter.next() {
                        Some(c) => c,
                        None => {
                            cx.emit_err(InvalidOnClause::Empty { span });
                            continue;
                        }
                    };

                    let condition = parse_condition(condition);

                    if items.len() < 2 {
                        // Something like `#[rustc_on_unimplemented(on(.., /* nothing */))]`
                        // There's a condition but no directive behind it, this is a mistake.
                        malformed!();
                    }

                    let mut directive =
                        or_malformed!(parse_directive_items(cx, mode, iter, false)?);

                    match condition {
                        Ok(c) => {
                            directive.condition = Some(c);
                            subcommands.push(directive);
                        }
                        Err(e) => {
                            cx.emit_err(e);
                        }
                    }
                } else {
                    malformed!();
                }
            }

            _other => {
                malformed!();
            }
        }
    }

    Some(OnUnimplementedDirective {
        is_rustc_attr: matches!(mode, Mode::RustcOnUnimplemented),
        condition,
        subcommands,
        message,
        label,
        notes,
        parent_label,
        append_const_msg,
    })
}
