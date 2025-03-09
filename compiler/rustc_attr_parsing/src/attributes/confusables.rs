use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use super::{AcceptMapping, AttributeParser};
use crate::context::{FinalizeContext, Stage};
use crate::session_diagnostics;

#[derive(Default)]
pub(crate) struct ConfusablesParser {
    confusables: ThinVec<Symbol>,
    first_span: Option<Span>,
}

impl<S: Stage> AttributeParser<S> for ConfusablesParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(&[sym::rustc_confusables], |this, cx, args| {
        let Some(list) = args.list() else {
            // FIXME(jdonszelmann): error when not a list? Bring validation code here.
            //       NOTE: currently subsequent attributes are silently ignored using
            //       tcx.get_attr().
            return;
        };

        if list.is_empty() {
            cx.emit_err(session_diagnostics::EmptyConfusables { span: cx.attr_span });
        }

        for param in list.mixed() {
            let span = param.span();

            let Some(lit) = param.lit() else {
                cx.emit_err(session_diagnostics::IncorrectMetaItem {
                    span,
                    suggestion: Some(session_diagnostics::IncorrectMetaItemSuggestion {
                        lo: span.shrink_to_lo(),
                        hi: span.shrink_to_hi(),
                    }),
                });
                continue;
            };

            this.confusables.push(lit.symbol);
        }

        this.first_span.get_or_insert(cx.attr_span);
    })];

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if self.confusables.is_empty() {
            return None;
        }

        Some(AttributeKind::Confusables {
            symbols: self.confusables,
            first_span: self.first_span.unwrap(),
        })
    }
}
