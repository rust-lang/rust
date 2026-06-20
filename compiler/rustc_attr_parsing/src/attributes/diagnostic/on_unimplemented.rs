use rustc_feature::AttributeStability;
use rustc_hir::attrs::diagnostic::Directive;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnUnimplementedParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnimplementedParser {
    fn parse<'sess>(&mut self, cx: &mut AcceptContext<'_, 'sess>, args: &ArgParser, mode: Mode) {
        let span = cx.attr_span;
        self.span = Some(span);

        let Some(items) = parse_list(cx, args, mode) else { return };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        };
    }
}

impl AttributeParser for OnUnimplementedParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (
            &[sym::diagnostic, sym::on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            AttributeStability::Stable,
            |this, cx, args| {
                this.parse(cx, args, Mode::DiagnosticOnUnimplemented);
            },
        ),
        (
            &[sym::rustc_on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            unstable!(
                rustc_attrs,
                "see `#[diagnostic::on_unimplemented]` for the stable equivalent of this attribute"
            ),
            |this, cx, args| {
                this.parse(cx, args, Mode::RustcOnUnimplemented);
            },
        ),
    ];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Trait)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(_span) = self.span {
            Some(AttributeKind::OnUnimplemented {
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
