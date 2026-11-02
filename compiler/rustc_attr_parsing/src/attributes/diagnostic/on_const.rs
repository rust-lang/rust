use rustc_attr_ir::diagnostic::Directive;
use rustc_feature::AttributeStability;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
#[derive(Default)]
pub(crate) struct OnConstParser {
    path_span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl AttributeParser for OnConstParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_const],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually below
        |this, cx, args| {
            gate_diagnostic_attr!(diagnostic_on_const);

            let path_span = cx.attr_path.span;
            this.path_span = Some(path_span);

            let mode = Mode::DiagnosticOnConst;

            let Some(items) = parse_list(cx, args, mode) else { return };

            let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) else {
                return;
            };
            merge_directives(cx, &mut this.directive, (path_span, directive));
        },
    )];

    // "Allowed" on all targets; noop on anything but non-const trait impls;
    // this linted on in parser.
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowListWarnRest(&[
        // FIXME(mejrs) no constness field on `Target`,
        // so non-constness is still checked in check_attr.rs
        Allow(Target::Impl { of_trait: true }),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(path_span) = self.path_span {
            Some(AttributeKind::OnConst {
                span: path_span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
