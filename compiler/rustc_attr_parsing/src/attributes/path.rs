use super::prelude::*;

pub(crate) struct PathParser;

impl<S: Stage> SingleAttributeParser<S> for PathParser {
    const PATH: &[Symbol] = &[sym::path];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Mod), Error(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "file",
        "https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(path) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::Path(path, cx.attr_span))
    }
}
