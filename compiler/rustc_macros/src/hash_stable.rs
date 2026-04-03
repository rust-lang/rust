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
        if !meta.path().is_ident("stable_hasher") {
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
            panic!("error parsing stable_hasher");
        }
    }
    attrs
}

pub(crate) fn hash_stable_derive(s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    hash_stable_derive_with_mode(s, HashStableMode::Normal)
}

pub(crate) fn hash_stable_no_context_derive(
    s: synstructure::Structure<'_>,
) -> proc_macro2::TokenStream {
    hash_stable_derive_with_mode(s, HashStableMode::NoContext)
}

enum HashStableMode {
    // Do a normal derive, where any generic type parameter gets a `HashStable` bound.
    // For example, in `struct Abc<T, U>(T, U)` the added bounds are `T: HashStable` and
    // `U: HashStable`.
    Normal,

    // Do an (almost-)perfect derive, where any field with a generic type parameter gets a
    // `HashStable` bound. For example, in `struct Def<T, U>(T, U::Assoc)` the added bounds are
    // `T::HashStable` and `U::Assoc: HashStable` (not `U: HashStable`).
    //
    // This is used most commonly in `rustc_type_ir` for types like `TyKind<I: Interner>`.
    // `Interner` does not impl `HashStable`, but the fields of `TyKind` do not use `I` itself,
    // instead only using associated types from `I` such as `I::Region`. On types like `TyKind` we
    // typically also see the use of `derive_where` for built-in traits such as `Debug`.
    NoContext,
}

fn hash_stable_derive_with_mode(
    mut s: synstructure::Structure<'_>,
    mode: HashStableMode,
) -> proc_macro2::TokenStream {
    let add_bounds = match mode {
        HashStableMode::Normal => synstructure::AddBounds::Generics,
        HashStableMode::NoContext => synstructure::AddBounds::Fields,
    };

    s.add_bounds(add_bounds);

    let discriminant = hash_stable_discriminant(&mut s);
    let body = hash_stable_body(&mut s);

    s.bound_impl(
        quote!(::rustc_data_structures::stable_hasher::HashStable),
        quote! {
            #[inline]
            fn hash_stable<__Hcx: ::rustc_data_structures::stable_hasher::HashStableContext>(
                &self,
                __hcx: &mut __Hcx,
                __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher
            ) {
                #discriminant
                match *self { #body }
            }
        },
    )
}

fn hash_stable_discriminant(s: &mut synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    match s.ast().data {
        syn::Data::Enum(_) => quote! {
            ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        },
        syn::Data::Struct(_) => quote! {},
        syn::Data::Union(_) => panic!("cannot derive on union"),
    }
}

fn hash_stable_body(s: &mut synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    s.each(|bi| {
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
    })
}
