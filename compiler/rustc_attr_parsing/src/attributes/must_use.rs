use rustc_errors::DiagArgValue;

use super::prelude::*;
use crate::session_diagnostics::IllFormedAttributeInputLint;

pub(crate) struct MustUseParser;

impl<S: Stage> SingleAttributeParser<S> for MustUseParser {
    const PATH: &[Symbol] = &[sym::must_use];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Union),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::ForeignFn),
        // `impl Trait` in return position can trip
        // `unused_must_use` if `Trait` is marked as
        // `#[must_use]`
        Allow(Target::Trait),
        Error(Target::WherePredicate),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word, NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::MustUse {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(value_str) = name_value.value_as_str() else {
                        cx.expected_string_literal(
                            name_value.value_span,
                            Some(&name_value.value_as_lit()),
                        );
                        return None;
                    };
                    Some(value_str)
                }
                ArgParser::List(_) => {
                    let suggestions = <Self as SingleAttributeParser<S>>::TEMPLATE
                        .suggestions(cx.attr_style, "must_use");
                    cx.emit_err(IllFormedAttributeInputLint {
                        num_suggestions: suggestions.len(),
                        suggestions: DiagArgValue::StrListSepByAnd(
                            suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                        ),
                        span: cx.attr_span,
                    });
                    return None;
                }
            },
        })
    }
}
