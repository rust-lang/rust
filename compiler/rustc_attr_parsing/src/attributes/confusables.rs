use rustc_attr_data_structures::AttributeKind;
use rustc_feature::template;
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
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::rustc_confusables],
        template!(List: r#""name1", "name2", ..."#),
        |this, cx, args| {
            let Some(list) = args.list() else {
                cx.expected_list(cx.attr_span);
                return;
            };

            if list.is_empty() {
                cx.emit_err(session_diagnostics::EmptyConfusables { span: cx.attr_span });
            }

            for param in list.mixed() {
                let span = param.span();

                let Some(lit) = param.lit().and_then(|i| i.value_str()) else {
                    cx.expected_string_literal(span, param.lit());
                    continue;
                };

                this.confusables.push(lit);
            }

            this.first_span.get_or_insert(cx.attr_span);
        },
    )];

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
