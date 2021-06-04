//! This module provides utilities for handling attributes on variants
//! of `config_type` enum. Currently there are two types of attributes
//! that could appear on the variants of `config_type` enum: `doc_hint`
//! and `value`. Both comes in the form of name-value pair whose value
//! is string literal.

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

/// Returns a string literal value if the given attribute is `value`
/// attribute or `None` otherwise.
pub fn config_value(attr: &syn::Attribute) -> Option<String> {
    get_name_value_str_lit(attr, "value")
}

/// Returns `true` if the given attribute is a `value` attribute.
pub fn is_config_value(attr: &syn::Attribute) -> bool {
    is_attr_name_value(attr, "value")
}

fn is_attr_name_value(attr: &syn::Attribute, name: &str) -> bool {
    attr.parse_meta().ok().map_or(false, |meta| match meta {
        syn::Meta::NameValue(syn::MetaNameValue { ref path, .. }) if path.is_ident(name) => true,
        _ => false,
    })
}

fn get_name_value_str_lit(attr: &syn::Attribute, name: &str) -> Option<String> {
    attr.parse_meta().ok().and_then(|meta| match meta {
        syn::Meta::NameValue(syn::MetaNameValue {
            ref path,
            lit: syn::Lit::Str(ref lit_str),
            ..
        }) if path.is_ident(name) => Some(lit_str.value()),
        _ => None,
    })
}
