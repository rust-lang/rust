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

fn no_hash_stable_eq(s: &synstructure::Structure<'_>) -> bool {
    for attr in &s.ast().attrs {
        if let Ok(meta) = attr.parse_meta() {
            if !meta.path().is_ident("stable_hasher") {
                continue;
            }
            let arg: syn::Ident = attr.parse_args().unwrap();
            if arg.to_string() == "no_hash_stable_eq" {
                return true;
            } else {
                panic!("Unexpected argument {:?}", arg);
            }
        }
    }
    return false;
}

pub fn hash_stable_generic_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    let orig = s.clone();
    let no_eq = no_hash_stable_eq(&orig);

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

    let impl_body = s.bound_impl(
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
    );

    if no_eq {
        impl_body
    } else {
        let eq_impl = hash_stable_eq_derive(orig);
        //println!("Eq impl:\n{}", eq_impl);
        quote! {
            #impl_body
            #eq_impl
        }
    }
}

pub fn hash_stable_eq_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    let mut other = s.clone();
    other.binding_name(|_bi, i| Ident::new(&format!("__binding_other_{}", i), proc_macro2::Span::call_site()));

    let eq_body: proc_macro2::TokenStream = s.variants().iter().zip(other.variants()).map(|(variant1, variant2)| {

        let first_pat = variant1.pat();
        let second_pat = variant2.pat();

        let compare = std::iter::once(quote! { true }).chain(variant1.bindings().iter().zip(variant2.bindings()).map(|(binding1, binding2)| {
            let attrs = parse_attributes(binding1.ast());
            if attrs.ignore {
                quote! { true }
            } else if let Some(project) = attrs.project {
                quote! {
                    ::rustc_data_structures::stable_hasher::HashStableEq::hash_stable_eq(
                        #binding1.#project, #binding2.#project
                    )
                }
            } else {
                quote! {
                    ::rustc_data_structures::stable_hasher::HashStableEq::hash_stable_eq(
                        #binding1, #binding2
                    )
                }
            }
        }));
        quote! {
            (#first_pat, #second_pat) => {
                #(#compare)&&*
            }
        }
    }).collect();

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bound_impl(
        quote!(::rustc_data_structures::stable_hasher::HashStableEq),
        quote! {
            fn hash_stable_eq(&self, other: &Self) -> bool {
                match (self, other) {
                    #eq_body
                    _ => false
                }
            }
        }
    )    
}

pub fn hash_stable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    let orig = s.clone();
    let no_eq = no_hash_stable_eq(&orig);

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

    let impl_body = s.bound_impl(
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
    );

    if no_eq {
        impl_body
    } else {
        let eq_impl = hash_stable_eq_derive(orig);
        quote! {
            #impl_body
            #eq_impl
        }
    }
}
