use rustc_feature::AttributeStability;
use rustc_hir::attrs::diagnostic::Directive;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnUnknownParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnknownParser {
    fn parse<'sess>(&mut self, cx: &mut AcceptContext<'_, 'sess>, args: &ArgParser, mode: Mode) {
        if let Some(features) = cx.features
            && !features.diagnostic_on_unknown()
        {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
            args.ignore_args();
            return;
        }
        let span = cx.attr_span;
        self.span = Some(span);

        let Some(items) = parse_list(cx, args, mode) else { return };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        };
    }
}

impl AttributeParser for OnUnknownParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::on_unknown],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        AttributeStability::Stable, // Unstable, stability checked manually in the parser
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnUnknown);
        },
    )];
    // "Allowed" for all targets, but noop for all but use statements.
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
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
