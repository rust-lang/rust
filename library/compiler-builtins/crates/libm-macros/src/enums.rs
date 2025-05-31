use heck::ToUpperCamelCase;
use proc_macro2 as pm2;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::spanned::Spanned;
use syn::{Fields, ItemEnum, Variant};

use crate::{ALL_OPERATIONS, base_name};

/// Implement `#[function_enum]`, see documentation in `lib.rs`.
pub fn function_enum(
    mut item: ItemEnum,
    attributes: pm2::TokenStream,
) -> syn::Result<pm2::TokenStream> {
    expect_empty_enum(&item)?;
    let attr_span = attributes.span();
    let mut attr = attributes.into_iter();

    // Attribute should be the identifier of the `BaseName` enum.
    let Some(tt) = attr.next() else {
        return Err(syn::Error::new(attr_span, "expected one attribute"));
    };

    let pm2::TokenTree::Ident(base_enum) = tt else {
        return Err(syn::Error::new(tt.span(), "expected an identifier"));
    };

    if let Some(tt) = attr.next() {
        return Err(syn::Error::new(
            tt.span(),
            "unexpected token after identifier",
        ));
    }

    let enum_name = &item.ident;
    let mut as_str_arms = Vec::new();
    let mut from_str_arms = Vec::new();
    let mut base_arms = Vec::new();

    for func in ALL_OPERATIONS.iter() {
        let fn_name = func.name;
        let ident = Ident::new(&fn_name.to_upper_camel_case(), Span::call_site());
        let bname_ident = Ident::new(&base_name(fn_name).to_upper_camel_case(), Span::call_site());

        // Match arm for `fn as_str(self)` matcher
        as_str_arms.push(quote! { Self::#ident => #fn_name });
        from_str_arms.push(quote! { #fn_name => Self::#ident });

        // Match arm for `fn base_name(self)` matcher
        base_arms.push(quote! { Self::#ident => #base_enum::#bname_ident });

        let variant = Variant {
            attrs: Vec::new(),
            ident,
            fields: Fields::Unit,
            discriminant: None,
        };

        item.variants.push(variant);
    }

    let variants = item.variants.iter();

    let res = quote! {
        // Instantiate the enum
        #item

        impl #enum_name {
            /// All variants of this enum.
            pub const ALL: &[Self] = &[
                #( Self::#variants, )*
            ];

            /// The stringified version of this function name.
            pub const fn as_str(self) -> &'static str {
                match self {
                    #( #as_str_arms , )*
                }
            }

            /// If `s` is the name of a function, return it.
            pub fn from_str(s: &str) -> Option<Self> {
                let ret = match s {
                    #( #from_str_arms , )*
                    _ => return None,
                };
                Some(ret)
            }

            /// The base name enum for this function.
            pub const fn base_name(self) -> #base_enum {
                match self {
                    #( #base_arms, )*
                }
            }

            /// Return information about this operation.
            pub fn math_op(self) -> &'static crate::op::MathOpInfo {
                crate::op::ALL_OPERATIONS.iter().find(|op| op.name == self.as_str()).unwrap()
            }
        }
    };

    Ok(res)
}

/// Implement `#[base_name_enum]`, see documentation in `lib.rs`.
pub fn base_name_enum(
    mut item: ItemEnum,
    attributes: pm2::TokenStream,
) -> syn::Result<pm2::TokenStream> {
    expect_empty_enum(&item)?;
    if !attributes.is_empty() {
        let sp = attributes.span();
        return Err(syn::Error::new(sp.span(), "no attributes expected"));
    }

    let mut base_names: Vec<_> = ALL_OPERATIONS
        .iter()
        .map(|func| base_name(func.name))
        .collect();
    base_names.sort_unstable();
    base_names.dedup();

    let item_name = &item.ident;
    let mut as_str_arms = Vec::new();

    for base_name in base_names {
        let ident = Ident::new(&base_name.to_upper_camel_case(), Span::call_site());

        // Match arm for `fn as_str(self)` matcher
        as_str_arms.push(quote! { Self::#ident => #base_name });

        let variant = Variant {
            attrs: Vec::new(),
            ident,
            fields: Fields::Unit,
            discriminant: None,
        };

        item.variants.push(variant);
    }

    let res = quote! {
        // Instantiate the enum
        #item

        impl #item_name {
            /// The stringified version of this base name.
            pub const fn as_str(self) -> &'static str {
                match self {
                    #( #as_str_arms ),*
                }
            }
        }
    };

    Ok(res)
}

/// Verify that an enum is empty, otherwise return an error
fn expect_empty_enum(item: &ItemEnum) -> syn::Result<()> {
    if !item.variants.is_empty() {
        Err(syn::Error::new(
            item.variants.span(),
            "expected an empty enum",
        ))
    } else {
        Ok(())
    }
}
