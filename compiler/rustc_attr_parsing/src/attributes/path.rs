use super::prelude::*;

pub(crate) struct PathParser;

impl SingleAttributeParser for PathParser {
    const PATH: &[Symbol] = &[sym::path];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Mod), Error(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "file",
        "https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let nv = cx.expect_name_value(args, cx.attr_span, None)?;
        let path = cx.expect_string_literal(nv)?;

        Some(AttributeKind::Path(path))
    }
}
