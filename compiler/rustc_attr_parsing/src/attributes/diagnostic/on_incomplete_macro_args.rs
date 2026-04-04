use rustc_hir::attrs::diagnostic::Directive;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnIncompleteMacroArgsParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl<S: Stage> AttributeParser<S> for OnIncompleteMacroArgsParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_incomplete_macro_args],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            if !cx.features().diagnostic_on_incomplete_macro_args() {
                return;
            }

            let span = cx.attr_span;
            this.span = Some(span);

            if !matches!(cx.target, Target::MacroDef) {
                return;
            }

            let items = match args {
                ArgParser::List(items) if items.len() != 0 => items,
                ArgParser::NoArgs | ArgParser::List(_) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MissingOptionsForOnIncompleteMacroArgs,
                        span,
                    );
                    return;
                }
                ArgParser::NameValue(_) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnIncompleteMacroArgsAttr { span },
                        span,
                    );
                    return;
                }
            };

            let Some(directive) = parse_directive_items(
                cx,
                Mode::DiagnosticOnIncompleteMacroArgs,
                items.mixed(),
                true,
            ) else {
                return;
            };
            merge_directives(cx, &mut this.directive, (span, directive));
        },
    )];

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnIncompleteMacroArgs {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
