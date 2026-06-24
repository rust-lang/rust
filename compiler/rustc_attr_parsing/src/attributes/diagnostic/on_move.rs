use rustc_feature::AttributeStability;
use rustc_hir::attrs::AttributeKind;
use rustc_span::sym;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::template;

#[derive(Default)]
pub(crate) struct OnMoveParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnMoveParser {
    fn parse<'sess>(&mut self, cx: &mut AcceptContext<'_, 'sess>, args: &ArgParser, mode: Mode) {
        if !cx.features().diagnostic_on_move() {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
            args.ignore_args();
            return;
        }

        let span = cx.attr_span;
        self.span = Some(span);

        let Some(items) = parse_list(cx, args, mode) else { return };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        }
    }
}
impl AttributeParser for OnMoveParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_move],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually in the parser
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnMove);
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
