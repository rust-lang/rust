use rustc_errors::Diagnostic;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_session::lint::builtin::MISPLACED_DIAGNOSTIC_ATTRIBUTES;

use crate::ShouldEmit;
use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::errors::DiagnosticOnUnknownOnlyForImports;

#[derive(Default)]
pub(crate) struct OnUnknownParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnknownParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        if let Some(features) = cx.features
            && !features.diagnostic_on_unknown()
        {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
            return;
        }
        let span = cx.attr_span;
        self.span = Some(span);

        // At early parsing we get passed `Target::Crate` regardless of the item we're on.
        // Only do target checking if we're late.
        let early = matches!(cx.stage.should_emit(), ShouldEmit::Nothing);

        if !early && !matches!(cx.target, Target::Use) {
            let target_span = cx.target_span;
            cx.emit_dyn_lint(
                MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                move |dcx, level, _| {
                    DiagnosticOnUnknownOnlyForImports { target_span }.into_diag(dcx, level)
                },
                span,
            );
            return;
        }

        let Some(items) = parse_list(cx, args, mode) else { return };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        };
    }
}

impl<S: Stage> AttributeParser<S> for OnUnknownParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_unknown],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnUnknown);
        },
    )];
    // "Allowed" for all targets, but noop for all but use statements.
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnUnknown {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
