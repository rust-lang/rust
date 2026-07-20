//! The IR of the `#[query_group]` macro.

use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{Ident, PatType, Path, spanned::Spanned};

use crate::Cycle;

pub(crate) struct TrackedQuery {
    pub(crate) trait_name: Ident,
    pub(crate) signature: syn::Signature,
    pub(crate) pat_and_tys: Vec<PatType>,
    pub(crate) invoke: Option<Path>,
    pub(crate) default: Option<syn::Block>,
    pub(crate) cycle: Option<Cycle>,
}

impl ToTokens for TrackedQuery {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &self.signature;
        let trait_name = &self.trait_name;

        let ret = &sig.output;

        let invoke = match &self.invoke {
            Some(path) => path.to_token_stream(),
            None => sig.ident.to_token_stream(),
        };

        let fn_ident = &sig.ident;
        let shim: Ident = format_ident!("{}_shim", fn_ident);

        let options = self
            .cycle
            .as_ref()
            .map(|Cycle { cycle_result }| {
                cycle_result.as_ref().map(|(ident, path)| quote!(#ident=#path))
            })
            .into_iter();
        let annotation = quote!(#[salsa_macros::tracked( #(#options),* )]);

        let pat_and_tys = &self.pat_and_tys;
        let params = self
            .pat_and_tys
            .iter()
            .map(|pat_type| pat_type.pat.clone())
            .collect::<Vec<Box<syn::Pat>>>();

        let invoke_block = match &self.default {
            Some(default) => quote! { #default },
            None => {
                let invoke_params: proc_macro2::TokenStream = quote! {db, #(#params),*};
                quote_spanned! { invoke.span() =>  {#invoke(#invoke_params)}}
            }
        };

        let method = quote! {
            #sig {
                #annotation
                fn #shim<'db>(
                    db: &'db dyn #trait_name,
                    #(#pat_and_tys),*
                ) #ret
                    #invoke_block

                #shim(self, #(#params),*)
            }
        };

        method.to_tokens(tokens);
    }
}

pub(crate) struct Transparent {
    pub(crate) signature: syn::Signature,
    pub(crate) pat_and_tys: Vec<PatType>,
    pub(crate) invoke: Option<Path>,
    pub(crate) default: Option<syn::Block>,
}

impl ToTokens for Transparent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &self.signature;

        let ty = self
            .pat_and_tys
            .iter()
            .map(|pat_type| pat_type.pat.clone())
            .collect::<Vec<Box<syn::Pat>>>();

        let invoke = match &self.invoke {
            Some(path) => path.to_token_stream(),
            None => sig.ident.to_token_stream(),
        };

        let method = match &self.default {
            Some(default) => quote! {
                #sig { let db = self; #default }
            },
            None => quote! {
                #sig {
                    #invoke(self, #(#ty),*)
                }
            },
        };

        method.to_tokens(tokens);
    }
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Queries {
    TrackedQuery(TrackedQuery),
    Transparent(Transparent),
}

impl ToTokens for Queries {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Queries::TrackedQuery(tracked_query) => tracked_query.to_tokens(tokens),
            Queries::Transparent(transparent) => transparent.to_tokens(tokens),
        }
    }
}
