use rustc_feature::template;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::sym;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

#[derive(Default)]
pub(crate) struct OnMoveParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnMoveParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        if !cx.features().diagnostic_on_move() {
            // `UnknownDiagnosticAttribute` is emitted in rustc_resolve/macros.rs
            args.ignore_args();
            return;
        }

        let span = cx.attr_span;
        self.span = Some(span);
        let Some(list) = args.list() else {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::MissingOptionsForOnMove,
                span,
            );
            return;
        };

        if list.is_empty() {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::OnMoveMalformedAttrExpectedLiteralOrDelimiter,
                list.span,
            );
            return;
        }

        if let Some(directive) = parse_directive_items(cx, mode, list.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        }
    }
}
impl<S: Stage> AttributeParser<S> for OnMoveParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_move],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnMove);
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnMove { span, directive: self.directive.map(|d| Box::new(d.1)) })
        } else {
            None
        }
    }
}
