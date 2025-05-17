use rustc_ast::attr::{AttributeExt, first_attr_value_str_by_name};
use rustc_attr_data_structures::RustcVersion;
use rustc_feature::is_builtin_attr_name;
use rustc_span::{Symbol, sym};

/// Parse a rustc version number written inside string literal in an attribute,
/// like appears in `since = "1.0.0"`. Suffixes like "-dev" and "-nightly" are
/// not accepted in this position, unlike when parsing CFG_RELEASE.
pub fn parse_version(s: Symbol) -> Option<RustcVersion> {
    let mut components = s.as_str().split('-');
    let d = components.next()?;
    if components.next().is_some() {
        return None;
    }
    let mut digits = d.splitn(3, '.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    let patch = digits.next().unwrap_or("0").parse().ok()?;
    Some(RustcVersion { major, minor, patch })
}

pub fn is_builtin_attr(attr: &impl AttributeExt) -> bool {
    attr.is_doc_comment() || attr.ident().is_some_and(|ident| is_builtin_attr_name(ident.name))
}

pub fn find_crate_name(attrs: &[impl AttributeExt]) -> Option<Symbol> {
    first_attr_value_str_by_name(attrs, sym::crate_name)
}

pub fn is_doc_alias_attrs_contain_symbol<'tcx, T: AttributeExt + 'tcx>(
    attrs: impl Iterator<Item = &'tcx T>,
    symbol: Symbol,
) -> bool {
    let doc_attrs = attrs.filter(|attr| attr.has_name(sym::doc));
    for attr in doc_attrs {
        let Some(values) = attr.meta_item_list() else {
            continue;
        };
        let alias_values = values.iter().filter(|v| v.has_name(sym::alias));
        for v in alias_values {
            if let Some(nested) = v.meta_item_list() {
                // #[doc(alias("foo", "bar"))]
                let mut iter = nested.iter().filter_map(|item| item.lit()).map(|item| item.symbol);
                if iter.any(|s| s == symbol) {
                    return true;
                }
            } else if let Some(meta) = v.meta_item()
                && let Some(lit) = meta.name_value_literal()
            {
                // #[doc(alias = "foo")]
                if lit.symbol == symbol {
                    return true;
                }
            }
        }
    }
    false
}
