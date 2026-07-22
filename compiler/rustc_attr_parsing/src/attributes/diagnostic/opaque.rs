use rustc_feature::AttributeStability;
use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Span, sym};

use crate::attributes::{AcceptMapping, AttributeParser};
use crate::context::{AcceptContext, FinalizeContext};
use crate::diagnostics::OpaqueDoesNotExpectArgs;
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;
use crate::{template, unstable};

#[derive(Default)]
pub(crate) struct OpaqueParser {
    attr_span: Option<Span>,
}

impl AttributeParser for OpaqueParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (
            &[sym::diagnostic, sym::opaque],
            template!(Word),
            AttributeStability::Stable, // Unstable, stability checked manually in the parser
            |this, cx, args| {
                if !cx.features().diagnostic_opaque() {
                    return;
                }
                this.parse(cx, args);
            },
        ),
        (
            // For use on exported macros, where using tool attributes is an error.
            &[sym::rustc_diagnostic_opaque],
            template!(Word),
            unstable!(
                rustc_attrs,
                "see `#[diagnostic::opaque]` for the nightly equivalent of this attribute"
            ),
            OpaqueParser::parse,
        ),
    ];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::MacroDef)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if let Some(_) = self.attr_span { Some(AttributeKind::Opaque) } else { None }
    }
}

impl OpaqueParser {
    fn parse<'sess>(&mut self, cx: &mut AcceptContext<'_, 'sess>, args: &ArgParser) {
        let attr_span = cx.attr_span;
        if let Some(earlier_span) = self.attr_span {
            cx.warn_unused_duplicate(earlier_span, attr_span);
        }
        self.attr_span = Some(attr_span);

        if !matches!(args, ArgParser::NoArgs) {
            cx.emit_lint(MALFORMED_DIAGNOSTIC_ATTRIBUTES, OpaqueDoesNotExpectArgs, attr_span);
        }
    }
}
