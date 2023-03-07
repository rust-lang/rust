use proc_macro2::{self, Ident};
use quote::quote;
use syn::{self, parse_quote, Meta, NestedMeta};

struct Attributes {
    ignore: bool,
    project: Option<Ident>,
}

fn parse_attributes(field: &syn::Field) -> Attributes {
    let mut attrs = Attributes { ignore: false, project: None };
    for attr in &field.attrs {
        if let Ok(meta) = attr.parse_meta() {
            if !meta.path().is_ident("stable_hasher") {
                continue;
            }
            let mut any_attr = false;
            if let Meta::List(list) = meta {
                for nested in list.nested.iter() {
                    if let NestedMeta::Meta(meta) = nested {
                        if meta.path().is_ident("ignore") {
                            attrs.ignore = true;
                            any_attr = true;
                        }
                        if meta.path().is_ident("project") {
                            if let Meta::List(list) = meta {
                                if let Some(NestedMeta::Meta(meta)) = list.nested.iter().next() {
                                    attrs.project = meta.path().get_ident().cloned();
                                    any_attr = true;
                                }
                            }
                        }
                    }
                }
            }
            if !any_attr {
                panic!("error parsing stable_hasher");
            }
        }
    }
    attrs
}

pub fn hash_stable_generic_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    let generic: syn::GenericParam = parse_quote!(__CTX);
    s.add_bounds(synstructure::AddBounds::Generics);
    s.add_impl_generic(generic);
    s.add_where_predicate(parse_quote! { __CTX: crate::HashStableContext });
    let body = s.each(|bi| {
        let attrs = parse_attributes(bi.ast());
        if attrs.ignore {
            quote! {}
        } else if let Some(project) = attrs.project {
            quote! {
                (&#bi.#project).hash_stable(__hcx, __hasher);
            }
        } else {
            quote! {
                #bi.hash_stable(__hcx, __hasher);
            }
        }
    });

    let discriminant = match s.ast().data {
        syn::Data::Enum(_) => quote! {
            ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        },
        syn::Data::Struct(_) => quote! {},
        syn::Data::Union(_) => panic!("cannot derive on union"),
    };

    s.bound_impl(
        quote!(::rustc_data_structures::stable_hasher::HashStable<__CTX>),
        quote! {
            #[inline]
            fn hash_stable(
                &self,
                __hcx: &mut __CTX,
                __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher) {
                #discriminant
                match *self { #body }
            }
        },
    )
}

pub fn hash_stable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    let generic: syn::GenericParam = parse_quote!('__ctx);
    s.add_bounds(synstructure::AddBounds::Generics);
    s.add_impl_generic(generic);
    let body = s.each(|bi| {
        let attrs = parse_attributes(bi.ast());
        if attrs.ignore {
            quote! {}
        } else if let Some(project) = attrs.project {
            quote! {
                (&#bi.#project).hash_stable(__hcx, __hasher);
            }
        } else {
            quote! {
                #bi.hash_stable(__hcx, __hasher);
            }
        }
    });

    let discriminant = match s.ast().data {
        syn::Data::Enum(_) => quote! {
            ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        },
        syn::Data::Struct(_) => quote! {},
        syn::Data::Union(_) => panic!("cannot derive on union"),
    };

    s.bound_impl(
        quote!(
            ::rustc_data_structures::stable_hasher::HashStable<
                ::rustc_query_system::ich::StableHashingContext<'__ctx>,
            >
        ),
        quote! {
            #[inline]
            fn hash_stable(
                &self,
                __hcx: &mut ::rustc_query_system::ich::StableHashingContext<'__ctx>,
                __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher) {
                #discriminant
                match *self { #body }
            }
        },
    )
}
