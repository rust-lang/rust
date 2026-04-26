use rustc_span::hygiene::Transparency;

use super::prelude::*;

pub(crate) struct RustcMacroTransparencyParser;

impl<S: Stage> SingleAttributeParser<S> for RustcMacroTransparencyParser {
    const PATH: &[Symbol] = &[sym::rustc_macro_transparency];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Custom(|cx, used, unused| {
        cx.dcx().span_err(vec![used, unused], "multiple macro transparency attributes");
    });
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::MacroDef)]);
    const TEMPLATE: AttributeTemplate =
        template!(NameValueStr: ["transparent", "semiopaque", "opaque"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let nv = cx.expect_name_value(args, cx.attr_span, None)?;
        match nv.value_as_str() {
            Some(sym::transparent) => Some(Transparency::Transparent),
            Some(sym::semiopaque) => Some(Transparency::SemiOpaque),
            Some(sym::opaque) => Some(Transparency::Opaque),
            Some(_) => {
                cx.adcx().expected_specific_argument_strings(
                    nv.value_span,
                    &[sym::transparent, sym::semiopaque, sym::opaque],
                );
                None
            }
            None => None,
        }
        .map(AttributeKind::RustcMacroTransparency)
    }
}
