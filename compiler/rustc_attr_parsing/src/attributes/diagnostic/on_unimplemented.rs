use rustc_hir::attrs::diagnostic::Directive;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;

use crate::attributes::diagnostic::*;
use crate::attributes::prelude::*;

#[derive(Default)]
pub(crate) struct OnUnimplementedParser {
    span: Option<Span>,
    directive: Option<(Span, Directive)>,
}

impl OnUnimplementedParser {
    fn parse<'sess, S: Stage>(
        &mut self,
        cx: &mut AcceptContext<'_, 'sess, S>,
        args: &ArgParser,
        mode: Mode,
    ) {
        let span = cx.attr_span;
        self.span = Some(span);

        // If target is not a trait, returning early will make `finalize` emit a
        // `AttributeKind::OnUnimplemented {span, directive: None }`, to prevent it being
        // accidentally used on non-trait items like trait aliases.
        if !matches!(cx.target, Target::Trait) {
            // Lint later emitted in check_attr
            return;
        }

        let items = match args {
            ArgParser::List(items) if items.len() != 0 => items,
            ArgParser::NoArgs | ArgParser::List(_) => {
                cx.emit_lint(
                    MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    AttributeLintKind::MissingOptionsForOnUnimplemented,
                    span,
                );
                return;
            }
            ArgParser::NameValue(_) => {
                cx.emit_lint(
                    MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    AttributeLintKind::MalformedOnUnimplementedAttr { span },
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

impl<S: Stage> AttributeParser<S> for OnUnimplementedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (
            &[sym::diagnostic, sym::on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            |this, cx, args| {
                this.parse(cx, args, Mode::DiagnosticOnUnimplemented);
            },
        ),
        (
            &[sym::rustc_on_unimplemented],
            template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
            |this, cx, args| {
                this.parse(cx, args, Mode::RustcOnUnimplemented);
            },
        ),
    ];
    //FIXME attribute is not parsed for non-traits but diagnostics are issued in `check_attr.rs`
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(span) = self.span {
            Some(AttributeKind::OnUnimplemented {
                span,
                directive: self.directive.map(|d| Box::new(d.1)),
            })
        } else {
            None
        }
    }
}
