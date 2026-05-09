use proc_macro2::Ident;
use quote::quote;

struct Attributes {
    ignore: bool,
    project: Option<Ident>,
}

fn parse_attributes(field: &syn::Field) -> Attributes {
    let mut attrs = Attributes { ignore: false, project: None };
    for attr in &field.attrs {
        let meta = &attr.meta;
        if !meta.path().is_ident("stable_hash") {
            continue;
        }
        let mut any_attr = false;
        let _ = attr.parse_nested_meta(|nested| {
            if nested.path.is_ident("ignore") {
                attrs.ignore = true;
                any_attr = true;
            }
            if nested.path.is_ident("project") {
                let _ = nested.parse_nested_meta(|meta| {
                    if attrs.project.is_none() {
                        attrs.project = meta.path.get_ident().cloned();
                    }
                    any_attr = true;
                    Ok(())
                });
            }
            Ok(())
        });
        if !any_attr {
            panic!("error parsing stable_hash");
        }
    }
    attrs
}

pub(crate) fn stable_hash_derive(s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    stable_hash_derive_with_mode(s, StableHashMode::Normal)
}

pub(crate) fn stable_hash_no_context_derive(
    s: synstructure::Structure<'_>,
) -> proc_macro2::TokenStream {
    stable_hash_derive_with_mode(s, StableHashMode::NoContext)
}

enum StableHashMode {
    // Do a normal derive, where any generic type parameter gets a `StableHash` bound.
    // For example, in `struct Abc<T, U>(T, U)` the added bounds are `T: StableHash` and
    // `U: StableHash`.
    Normal,

    // Do an (almost-)perfect derive, where any field with a generic type parameter gets a
    // `StableHash` bound. For example, in `struct Def<T, U>(T, U::Assoc)` the added bounds are
    // `T::StableHash` and `U::Assoc: StableHash` (not `U: StableHash`).
    //
    // This is used most commonly in `rustc_type_ir` for types like `TyKind<I: Interner>`.
    // `Interner` does not impl `StableHash`, but the fields of `TyKind` do not use `I` itself,
    // instead only using associated types from `I` such as `I::Region`. On types like `TyKind` we
    // typically also see the use of `derive_where` for built-in traits such as `Debug`.
    NoContext,
}

fn stable_hash_derive_with_mode(
    mut s: synstructure::Structure<'_>,
    mode: StableHashMode,
) -> proc_macro2::TokenStream {
    let add_bounds = match mode {
        StableHashMode::Normal => synstructure::AddBounds::Generics,
        StableHashMode::NoContext => synstructure::AddBounds::Fields,
    };

    s.add_bounds(add_bounds);

    let discriminant = stable_hash_discriminant(&mut s);
    let body = stable_hash_body(&mut s);

    s.bound_impl(
        quote!(::rustc_data_structures::stable_hash::StableHash),
        quote! {
            #[inline]
            fn stable_hash<__Hcx: ::rustc_data_structures::stable_hash::StableHashCtxt>(
                &self,
                __hcx: &mut __Hcx,
                __hasher: &mut ::rustc_data_structures::stable_hash::StableHasher
            ) {
                #discriminant
                match *self { #body }
            }
        },
    )
}

fn stable_hash_discriminant(s: &mut synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    match s.ast().data {
        syn::Data::Enum(_) => quote! {
            ::std::mem::discriminant(self).stable_hash(__hcx, __hasher);
        },
        syn::Data::Struct(_) => quote! {},
        syn::Data::Union(_) => panic!("cannot derive on union"),
    }
}

fn stable_hash_body(s: &mut synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    s.each(|bi| {
        let attrs = parse_attributes(bi.ast());
        if attrs.ignore {
            quote! {}
        } else if let Some(project) = attrs.project {
            quote! {
                (&#bi.#project).stable_hash(__hcx, __hasher);
            }
        } else {
            quote! {
                #bi.stable_hash(__hcx, __hasher);
            }
        }
    })
}
