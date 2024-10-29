use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::Data;
use syn::spanned::Spanned;
use synstructure::Structure;

pub(crate) fn try_from_u32(s: Structure<'_>) -> TokenStream {
    let span_error = |span, message: &str| {
        quote_spanned! { span => const _: () = ::core::compile_error!(#message); }
    };

    // Must be applied to an enum type.
    if let Some(span) = match &s.ast().data {
        Data::Enum(_) => None,
        Data::Struct(s) => Some(s.struct_token.span()),
        Data::Union(u) => Some(u.union_token.span()),
    } {
        return span_error(span, "type is not an enum (TryFromU32)");
    }

    // The enum's variants must not have fields.
    let variant_field_errors = s
        .variants()
        .iter()
        .filter_map(|v| v.ast().fields.iter().map(|f| f.span()).next())
        .map(|span| span_error(span, "enum variant cannot have fields (TryFromU32)"))
        .collect::<TokenStream>();
    if !variant_field_errors.is_empty() {
        return variant_field_errors;
    }

    let ctor = s
        .variants()
        .iter()
        .map(|v| v.construct(|_, _| -> TokenStream { unreachable!() }))
        .collect::<Vec<_>>();
    // FIXME(edition_2024): Fix the `keyword_idents_2024` lint to not trigger here?
    #[allow(keyword_idents_2024)]
    s.gen_impl(quote! {
        // The surrounding code might have shadowed these identifiers.
        use ::core::convert::TryFrom;
        use ::core::primitive::u32;
        use ::core::result::Result::{self, Ok, Err};

        gen impl TryFrom<u32> for @Self {
            type Error = u32;

            #[allow(deprecated)] // Don't warn about deprecated variants.
            fn try_from(value: u32) -> Result<Self, Self::Error> {
                #( if value == const { #ctor as u32 } { return Ok(#ctor) } )*
                Err(value)
            }
        }
    })
}
