use super::prelude::*;
use crate::session_diagnostics::EmptyConfusables;

#[derive(Default)]
pub(crate) struct ConfusablesParser {
    confusables: ThinVec<Symbol>,
    first_span: Option<Span>,
}

impl AttributeParser for ConfusablesParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::rustc_confusables],
        template!(List: &[r#""name1", "name2", ..."#]),
        |this, cx, args| {
            let Some(list) = cx.expect_list(args, cx.attr_span) else { return };

            if list.is_empty() {
                cx.emit_err(EmptyConfusables { span: cx.attr_span });
            }

            for param in list.mixed() {
                let Some(lit) = cx.expect_string_literal(param) else {
                    continue;
                };

                this.confusables.push(lit);
            }

            this.first_span.get_or_insert(cx.attr_span);
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Method(MethodKind::Inherent))]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        if self.confusables.is_empty() {
            return None;
        }

        Some(AttributeKind::RustcConfusables { confusables: self.confusables })
    }
}
