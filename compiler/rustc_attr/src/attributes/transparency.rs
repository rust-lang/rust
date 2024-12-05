use rustc_hir::AttributeKind;
use rustc_span::hygiene::Transparency;
use rustc_span::sym;

use super::SingleAttributeGroup;
use crate::parser::{ArgParser, NameValueParser};

pub(crate) struct TransparencyGroup;

// TODO: fix this but I don't want to rn
#[allow(rustc::untranslatable_diagnostic)]
#[allow(rustc::diagnostic_outside_of_impl)]
impl SingleAttributeGroup for TransparencyGroup {
    const PATH: &'static [rustc_span::Symbol] = &[sym::rustc_macro_transparency];

    fn on_duplicate(cx: &crate::context::AttributeAcceptContext<'_>, first_span: rustc_span::Span) {
        cx.dcx().span_err(vec![first_span, cx.attr_span], "multiple macro transparency attributes");
    }

    fn convert(
        cx: &crate::context::AttributeAcceptContext<'_>,
        args: &crate::parser::GenericArgParser<'_, rustc_ast::Expr>,
    ) -> Option<AttributeKind> {
        match args.name_value().and_then(|nv| nv.value_as_str()) {
            Some(sym::transparent) => Some(Transparency::Transparent),
            Some(sym::semitransparent) => Some(Transparency::SemiTransparent),
            Some(sym::opaque) => Some(Transparency::Opaque),
            Some(other) => {
                cx.dcx().span_err(cx.attr_span, format!("unknown macro transparency: `{other}`"));
                None
            }
            None => None,
        }
        .map(AttributeKind::MacroTransparency)
    }
}
