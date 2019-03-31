pub fn find_doc_hint(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().filter_map(doc_hint).next()
}

pub fn is_doc_hint(attr: &syn::Attribute) -> bool {
    is_attr_name_value(attr, "doc_hint")
}

pub fn doc_hint(attr: &syn::Attribute) -> Option<String> {
    get_name_value_str_lit(attr, "doc_hint")
}

pub fn find_config_value(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().filter_map(config_value).next()
}

pub fn config_value(attr: &syn::Attribute) -> Option<String> {
    get_name_value_str_lit(attr, "value")
}

pub fn is_config_value(attr: &syn::Attribute) -> bool {
    is_attr_name_value(attr, "value")
}

fn is_attr_name_value(attr: &syn::Attribute, name: &str) -> bool {
    attr.interpret_meta().map_or(false, |meta| match meta {
        syn::Meta::NameValue(syn::MetaNameValue { ref ident, .. }) if ident == name => true,
        _ => false,
    })
}

fn get_name_value_str_lit(attr: &syn::Attribute, name: &str) -> Option<String> {
    attr.interpret_meta().and_then(|meta| match meta {
        syn::Meta::NameValue(syn::MetaNameValue {
            ref ident,
            lit: syn::Lit::Str(ref lit_str),
            ..
        }) if ident == name => Some(lit_str.value()),
        _ => None,
    })
}
