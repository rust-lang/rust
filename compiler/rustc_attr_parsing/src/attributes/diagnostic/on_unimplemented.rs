use rustc_errors::Diagnostic;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_session::lint::builtin::MISPLACED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::errors::DiagnosticOnUnimplementedOnlyForTraits;

#[derive(Default)]
pub(crate) struct OnUnimplementedParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnimplementedParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        let span = cx.attr_span;
        self.span = Some(span);

        if !matches!(cx.target, Target::Trait) {
            cx.emit_dyn_lint(
                MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                move |dcx, level, _| DiagnosticOnUnimplementedOnlyForTraits.into_diag(dcx, level),
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

impl<S: Stage> AttributeParser<S> for OnUnimplementedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (
            &[sym::diagnostic, sym::on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            |this, cx, args| {
                this.parse(cx, args, Mode::DiagnosticOnUnimplemented);
            },
        ),
        (
            &[sym::rustc_on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            |this, cx, args| {
                this.parse(cx, args, Mode::RustcOnUnimplemented);
            },
        ),
    ];
    //FIXME attribute is not parsed for non-traits but diagnostics are issued in `check_attr.rs`
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnUnimplemented {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
