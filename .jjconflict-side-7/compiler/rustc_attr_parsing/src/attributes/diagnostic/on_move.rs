use rustc_attr_ir::AttributeKind;
use rustc_feature::AttributeStability;
use rustc_span::sym;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::target_checking::AllowedTargets;
use crate::template;

#[derive(Default)]
pub(crate) struct OnMoveParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl AttributeParser for OnMoveParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_move],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually below
        |this, cx, args| {
            gate_diagnostic_attr!(diagnostic_on_move);

            let span = cx.attr_span;
            this.span = Some(span);
            let mode = Mode::DiagnosticOnMove;

            let Some(items) = parse_list(cx, args, mode) else { return };

            if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
                merge_directives(cx, &mut this.directive, (span, directive));
            }
        },
    )];

    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Union),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(_span) = self.span {
            Some(AttributeKind::OnMove { directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
