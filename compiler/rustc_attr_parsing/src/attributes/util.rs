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
