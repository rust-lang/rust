use super::prelude::*;

pub(crate) struct PathParser;

impl<S: Stage> SingleAttributeParser<S> for PathParser {
    const PATH: &[Symbol] = &[sym::path];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Mod), Error(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "file",
        "https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let nv = cx.expect_name_value(args, cx.attr_span, None)?;
        let Some(path) = nv.value_as_str() else {
            cx.adcx().expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::Path(path, cx.attr_span))
    }
}
