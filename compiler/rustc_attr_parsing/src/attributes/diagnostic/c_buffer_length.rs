use rustc_attr_ir::AttributeKind;
use rustc_attr_ir::target::Target;
use rustc_feature::AttributeStability;
use rustc_lint_defs::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Ident, Span, sym};

use crate::attributes::diagnostic::gate_diagnostic_attr;
use crate::attributes::{AcceptMapping, AttributeParser};
use crate::context::FinalizeContext;
use crate::diagnostics::InvalidCBufferLength;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;
use crate::template;

#[derive(Default)]
pub(crate) struct CBufferLengthParser {
    args: Option<(Ident, Ident)>,
    span: Option<Span>,
}

impl AttributeParser for CBufferLengthParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[(
        &[sym::diagnostic, sym::c_buffer_length],
        template!(List: &["pointer, length"]),
        AttributeStability::Stable, // Unstable, stability checked manually below.
        |this, cx, args| {
            gate_diagnostic_attr!(diagnostic_c_buffer_length);
            let attr_span = cx.attr_span;

            let parsed = (|| {
                let list = args.as_list()?;
                if list.len() != 2 {
                    return None;
                }
                let mut items = list.mixed();
                let pointer = items.next()?.meta_item_no_args()?.ident()?;
                let length = items.next()?.meta_item_no_args()?.ident()?;
                (pointer.name != length.name).then_some((pointer, length))
            })();
            let Some(parsed) = parsed else {
                cx.emit_lint(MALFORMED_DIAGNOSTIC_ATTRIBUTES, InvalidCBufferLength, attr_span);
                return;
            };

            if let Some(span) = this.span {
                cx.warn_unused_duplicate(span, attr_span);
            } else {
                this.args = Some(parsed);
                this.span = Some(attr_span);
            }
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        self.args.map(|(pointer, length)| AttributeKind::CBufferLength { pointer, length })
    }
}
