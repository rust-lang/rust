use rustc_ast::attr::AttributeExt;
use rustc_attr_data_structures::TransparencyError;
use rustc_span::hygiene::Transparency;
use rustc_span::sym;

pub fn find_transparency(
    attrs: &[impl AttributeExt],
    macro_rules: bool,
) -> (Transparency, Option<TransparencyError>) {
    let mut transparency = None;
    let mut error = None;
    for attr in attrs {
        if attr.has_name(sym::rustc_macro_transparency) {
            if let Some((_, old_span)) = transparency {
                error = Some(TransparencyError::MultipleTransparencyAttrs(old_span, attr.span()));
                break;
            } else if let Some(value) = attr.value_str() {
                transparency = Some((
                    match value {
                        sym::transparent => Transparency::Transparent,
                        sym::semitransparent => Transparency::SemiTransparent,
                        sym::opaque => Transparency::Opaque,
                        _ => {
                            error =
                                Some(TransparencyError::UnknownTransparency(value, attr.span()));
                            continue;
                        }
                    },
                    attr.span(),
                ));
            }
        }
    }
    let fallback = if macro_rules { Transparency::SemiTransparent } else { Transparency::Opaque };
    (transparency.map_or(fallback, |t| t.0), error)
}
