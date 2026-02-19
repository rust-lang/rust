use rustc_hir::attrs::diagnostic::Directive;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;

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

            let items = match args {
                ArgParser::List(items) if items.len() != 0 => items,
                ArgParser::NoArgs | ArgParser::List(_) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MissingOptionsForOnConst,
                        span,
                    );
                    return;
                }
                ArgParser::NameValue(_) => {
                    cx.emit_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        AttributeLintKind::MalformedOnConstAttr { span },
                        span,
                    );
                    return;
                }
            };

            let Some(directive) =
                parse_directive_items(cx, Mode::DiagnosticOnConst, items.mixed(), true)
            else {
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
