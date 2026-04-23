use rustc_errors::Diagnostic;
use rustc_feature::template;
use rustc_hir::attrs::AttributeKind;
use rustc_session::lint::builtin::MISPLACED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::sym;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::context::{AcceptContext, Stage};
use crate::errors::DiagnosticOnMoveOnlyForAdt;
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

#[derive(Default)]
pub(crate) struct OnMoveParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnMoveParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        if !cx.features().diagnostic_on_move() {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
            return;
        }

        let span = cx.attr_span;
        self.span = Some(span);

        if !matches!(cx.target, Target::Enum | Target::Struct | Target::Union) {
            cx.emit_dyn_lint(
                MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                move |dcx, level, _| DiagnosticOnMoveOnlyForAdt.into_diag(dcx, level),
                span,
            );
            return;
        }

        let Some(items) = parse_list(cx, args, mode) else { return };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        }
    }
}
impl<S: Stage> AttributeParser<S> for OnMoveParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_move],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnMove);
        },
    )];

    // "Allowed" for all targets but noop if used on not-adt.
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnMove { span, directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
