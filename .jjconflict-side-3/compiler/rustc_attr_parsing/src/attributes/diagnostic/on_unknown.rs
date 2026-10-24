use rustc_attr_ir::diagnostic::Directive;
use rustc_feature::AttributeStability;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnUnknownParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl AttributeParser for OnUnknownParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_unknown],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually below
        |this, cx, args| {
            gate_diagnostic_attr!(diagnostic_on_unknown);

            let span = cx.attr_span;
            this.span = Some(span);
            let mode = Mode::DiagnosticOnUnknown;

            let Some(items) = parse_list(cx, args, mode) else { return };

            if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
                merge_directives(cx, &mut this.directive, (span, directive));
            };
        },
    )];
    // "Allowed" for all targets, but noop for all but use statements.
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Use),
        Allow(Target::Mod),
        Allow(Target::Crate),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(_span) = self.span {
            Some(AttributeKind::OnUnknown { directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
