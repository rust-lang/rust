use rustc_hir::attrs::diagnostic::Directive;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnConstParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl<S: Stage> AttributeParser<S> for OnConstParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_const],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            if !cx.features().diagnostic_on_const() {
                return;
            }

            let span = cx.attr_span;
            this.span = Some(span);
            let mode = Mode::DiagnosticOnConst;

            let Some(items) = parse_list(cx, args, mode) else { return };

            let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) else {
                return;
            };
            merge_directives(cx, &mut this.directive, (span, directive));
        },
    )];

    //FIXME Still checked in `check_attr.rs`
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnConst { span, directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
