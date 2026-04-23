use rustc_errors::Diagnostic;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_session::lint::builtin::MISPLACED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::errors::DiagnosticOnUnmatchArgsOnlyForMacros;

#[derive(Default)]
pub(crate) struct OnUnmatchArgsParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl<S: Stage> AttributeParser<S> for OnUnmatchArgsParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_unmatch_args],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            if !cx.features().diagnostic_on_unmatch_args() {
                return;
            }

            let span = cx.attr_span;
            this.span = Some(span);

            if !matches!(cx.target, Target::MacroDef) {
                cx.emit_dyn_lint(
                    MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                    move |dcx, level, _| DiagnosticOnUnmatchArgsOnlyForMacros.into_diag(dcx, level),
                    span,
                );
                return;
            }

            let mode = Mode::DiagnosticOnUnmatchArgs;
            let Some(items) = parse_list(cx, args, mode) else { return };

            let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) else {
                return;
            };
            merge_directives(cx, &mut this.directive, (span, directive));
        },
    )];

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnUnmatchArgs {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
