use rustc_errors::Diagnostic;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_session::lint::builtin::MISPLACED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::errors::DiagnosticOnConstOnlyForTraitImpls;
#[derive(Default)]
pub(crate) struct OnConstParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl AttributeParser for OnConstParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_const],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            if !cx.features().diagnostic_on_const() {
                // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
                return;
            }

            let span = cx.attr_span;
            this.span = Some(span);

            // FIXME(mejrs) no constness field on `Target`,
            // so non-constness is still checked in check_attr.rs
            if !matches!(cx.target, Target::Impl { of_trait: true }) {
                let target_span = cx.target_span;
                cx.emit_dyn_lint(
                    MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                    move |dcx, level| {
                        DiagnosticOnConstOnlyForTraitImpls { target_span }.into_diag(dcx, level)
                    },
                    span,
                );
                return;
            }

            let mode = Mode::DiagnosticOnConst;

            let Some(items) = parse_list(cx, args, mode) else { return };

            let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) else {
                return;
            };
            merge_directives(cx, &mut this.directive, (span, directive));
        },
    )];

    // "Allowed" on all targets; noop on anything but non-const trait impls;
    // this linted on in parser.
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnConst { span, directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
