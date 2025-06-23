use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::hygiene::Transparency;
use rustc_span::{Symbol, sym};

use super::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct TransparencyParser;

// FIXME(jdonszelmann): make these proper diagnostics
#[allow(rustc::untranslatable_diagnostic)]
#[allow(rustc::diagnostic_outside_of_impl)]
impl<S: Stage> SingleAttributeParser<S> for TransparencyParser {
    const PATH: &[Symbol] = &[sym::rustc_macro_transparency];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Custom(|cx, used, unused| {
        cx.dcx().span_err(vec![used, unused], "multiple macro transparency attributes");
    });
    const TEMPLATE: AttributeTemplate =
        template!(NameValueStr: "transparent|semitransparent|opaque");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        match nv.value_as_str() {
            Some(sym::transparent) => Some(Transparency::Transparent),
            Some(sym::semiopaque | sym::semitransparent) => Some(Transparency::SemiOpaque),
            Some(sym::opaque) => Some(Transparency::Opaque),
            Some(_) => {
                cx.expected_specific_argument_strings(
                    nv.value_span,
                    vec!["transparent", "semitransparent", "opaque"],
                );
                None
            }
            None => None,
        }
        .map(AttributeKind::MacroTransparency)
    }
}
