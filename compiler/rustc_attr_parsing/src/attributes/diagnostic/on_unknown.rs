use rustc_hir::attrs::diagnostic::Directive;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnUnknownParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnknownParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        if !cx.features().diagnostic_on_unknown() {
            args.ignore_args();
            return;
        }
        let span = cx.attr_span;
        self.span = Some(span);

        let items = match args {
            ArgParser::List(items) if !items.is_empty() => items,
            ArgParser::NoArgs | ArgParser::List(_) => {
                cx.emit_lint(
                    MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    AttributeLintKind::MissingOptionsForOnUnknown,
                    span,
                );
                return;
            }
            ArgParser::NameValue(_) => {
                cx.emit_lint(
                    MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    AttributeLintKind::MalformedOnUnknownAttr { span },
                    span,
                );
                return;
            }
        };

        if let Some(directive) = parse_directive_items(cx, mode, items.mixed(), true) {
            merge_directives(cx, &mut self.directive, (span, directive));
        };
    }
}

impl<S: Stage> AttributeParser<S> for OnUnknownParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::diagnostic, sym::on_unknown],
        template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
        |this, cx, args| {
            this.parse(cx, args, Mode::DiagnosticOnUnknown);
        },
    )];
    //FIXME attribute is not parsed for non-use statements but diagnostics are issued in `check_attr.rs`
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnUnknown {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
