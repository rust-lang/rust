use super::prelude::*;
pub(crate) struct CfiEncodingParser;
impl<S: Stage> SingleAttributeParser<S> for CfiEncodingParser {
    const PATH: &[Symbol] = &[sym::cfi_encoding];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Struct),
        Allow(Target::ForeignTy),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "encoding");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(name_value) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, Some(sym::cfi_encoding));
            return None;
        };

        let Some(value_str) = name_value.value_as_str() else {
            cx.expected_string_literal(name_value.value_span, None);
            return None;
        };

        if value_str.as_str().trim().is_empty() {
            cx.expected_non_empty_string_literal(name_value.value_span);
            return None;
        }

        Some(AttributeKind::CfiEncoding { encoding: value_str })
    }
}
