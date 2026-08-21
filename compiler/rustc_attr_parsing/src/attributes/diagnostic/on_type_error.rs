use rustc_attr_ir::AttributeKind;
use rustc_span::sym;

use crate::attributes::AttributeStability;
use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::target_checking::AllowedTargets;
use crate::template;

#[derive(Default)]
pub(crate) struct OnTypeErrorParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl AttributeParser for OnTypeErrorParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_type_error],
        template!(List: &[r#"note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually below
        |this, cx, args| {
            gate_diagnostic_attr!(diagnostic_on_type_error);

            let span = cx.attr_span;
            this.span = Some(span);
            let mode = Mode::DiagnosticOnTypeError;
            let Some(items) = parse_list(cx, args, mode) else { return };

            if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
                merge_directives(cx, &mut this.directive, (span, directive));
            }
        },
    )];

    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Enum),
        Allow(Target::Struct { has_default_field_values: false }),
        Allow(Target::Struct { has_default_field_values: true }),
        Allow(Target::Union),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnTypeError {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
