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
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "encoding");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let name_value = cx.expect_name_value(args, cx.attr_span, Some(sym::cfi_encoding))?;

        let Some(value_str) = name_value.value_as_str() else {
            cx.adcx().expected_string_literal(name_value.value_span, None);
            return None;
        };

        if value_str.as_str().trim().is_empty() {
            cx.adcx().expected_non_empty_string_literal(name_value.value_span);
            return None;
        }

        Some(AttributeKind::CfiEncoding { encoding: value_str })
    }
}
