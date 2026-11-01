//! This module provides utilities for handling attributes on variants
//! of `config_type` enum. Currently there are the following attributes
//! that could appear on the variants of `config_type` enum:
//!
//! - `doc_hint`: name-value pair whose value is string literal
//! - `value`: name-value pair whose value is string literal
//! - `unstable_variant`: name only

/// Returns the value of the first `doc_hint` attribute in the given slice or
/// `None` if `doc_hint` attribute is not available.
pub fn find_doc_hint(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().filter_map(doc_hint).next()
}

/// Returns `true` if the given attribute is a `doc_hint` attribute.
pub fn is_doc_hint(attr: &syn::Attribute) -> bool {
    is_attr_name_value(attr, "doc_hint")
}

/// Returns a string literal value if the given attribute is `doc_hint`
/// attribute or `None` otherwise.
pub fn doc_hint(attr: &syn::Attribute) -> Option<String> {
    get_name_value_str_lit(attr, "doc_hint")
}

/// Returns the value of the first `value` attribute in the given slice or
/// `None` if `value` attribute is not available.
pub fn find_config_value(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().filter_map(config_value).next()
}

/// Returns `true` if the there is at least one `unstable` attribute in the given slice.
pub fn any_unstable_variant(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(is_unstable_variant)
}

/// Returns a string literal value if the given attribute is `value`
/// attribute or `None` otherwise.
pub fn config_value(attr: &syn::Attribute) -> Option<String> {
    get_name_value_str_lit(attr, "value")
}

/// Returns `true` if the given attribute is a `value` attribute.
pub fn is_config_value(attr: &syn::Attribute) -> bool {
    is_attr_name_value(attr, "value")
}

/// Returns `true` if the given attribute is an `unstable` attribute.
pub fn is_unstable_variant(attr: &syn::Attribute) -> bool {
    is_attr_path(attr, "unstable_variant")
}

fn is_attr_name_value(attr: &syn::Attribute, name: &str) -> bool {
    match &attr.meta {
        syn::Meta::NameValue(syn::MetaNameValue { path, .. }) if path.is_ident(name) => true,
        _ => false,
    }
}

fn is_attr_path(attr: &syn::Attribute, name: &str) -> bool {
    match &attr.meta {
        syn::Meta::Path(path) if path.is_ident(name) => true,
        _ => false,
    }
}

fn get_name_value_str_lit(attr: &syn::Attribute, name: &str) -> Option<String> {
    match &attr.meta {
        syn::Meta::NameValue(syn::MetaNameValue {
            path,
            value:
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit_str),
                    ..
                }),
            ..
        }) if path.is_ident(name) => Some(lit_str.value()),
        _ => None,
    }
}
