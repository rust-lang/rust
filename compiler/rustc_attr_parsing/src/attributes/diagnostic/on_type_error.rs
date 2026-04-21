use rustc_feature::template;
use rustc_hir::attrs::AttributeKind;
use rustc_span::sym;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

#[derive(Default)]
pub(crate) struct OnTypeErrorParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnTypeErrorParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        if !cx.features().diagnostic_on_type_error() {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
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

impl<S: Stage> AttributeParser<S> for OnTypeErrorParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_type_error],
        template!(List: &[r#" note = "...""#]),
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnTypeError);
        },
    )];

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
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
