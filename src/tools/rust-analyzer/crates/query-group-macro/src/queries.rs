//! The IR of the `#[query_group]` macro.

use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{FnArg, Ident, PatType, Path, Receiver, ReturnType, Type, parse_quote, spanned::Spanned};

use crate::Cycle;

pub(crate) struct TrackedQuery {
    pub(crate) trait_name: Ident,
    pub(crate) signature: syn::Signature,
    pub(crate) pat_and_tys: Vec<PatType>,
    pub(crate) invoke: Option<Path>,
    pub(crate) default: Option<syn::Block>,
    pub(crate) cycle: Option<Cycle>,
    pub(crate) lru: Option<u32>,
    pub(crate) generated_struct: Option<GeneratedInputStruct>,
}

pub(crate) struct GeneratedInputStruct {
    pub(crate) input_struct_name: Ident,
    pub(crate) create_data_ident: Ident,
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
            .map(|Cycle { cycle_fn, cycle_initial, cycle_result }| {
                let cycle_fn = cycle_fn.as_ref().map(|(ident, path)| quote!(#ident=#path));
                let cycle_initial =
                    cycle_initial.as_ref().map(|(ident, path)| quote!(#ident=#path));
                let cycle_result = cycle_result.as_ref().map(|(ident, path)| quote!(#ident=#path));
                let options = cycle_fn.into_iter().chain(cycle_initial).chain(cycle_result);
                quote!(#(#options),*)
            })
            .into_iter()
            .chain(self.lru.map(|lru| quote!(lru = #lru)));
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

        let method = match &self.generated_struct {
            Some(generated_struct) => {
                let input_struct_name = &generated_struct.input_struct_name;
                let create_data_ident = &generated_struct.create_data_ident;

                quote! {
                    #sig {
                        #annotation
                        fn #shim<'db>(
                            db: &'db dyn #trait_name,
                            _input: #input_struct_name,
                            #(#pat_and_tys),*
                        ) #ret
                            #invoke_block
                        #shim(self, #create_data_ident(self), #(#params),*)
                    }
                }
            }
            None => {
                quote! {
                    #sig {
                        #annotation
                        fn #shim<'db>(
                            db: &'db dyn #trait_name,
                            #(#pat_and_tys),*
                        ) #ret
                            #invoke_block

                        #shim(self, #(#params),*)
                    }
                }
            }
        };

        method.to_tokens(tokens);
    }
}

pub(crate) struct InputQuery {
    pub(crate) signature: syn::Signature,
    pub(crate) create_data_ident: Ident,
}

impl ToTokens for InputQuery {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &self.signature;
        let fn_ident = &sig.ident;
        let create_data_ident = &self.create_data_ident;

        let method = quote! {
            #sig {
                let data = #create_data_ident(self);
                data.#fn_ident(self).unwrap()
            }
        };
        method.to_tokens(tokens);
    }
}

pub(crate) struct InputSetter {
    pub(crate) signature: syn::Signature,
    pub(crate) return_type: syn::Type,
    pub(crate) create_data_ident: Ident,
}

impl ToTokens for InputSetter {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &mut self.signature.clone();

        let ty = &self.return_type;
        let fn_ident = &sig.ident;
        let create_data_ident = &self.create_data_ident;

        let setter_ident = format_ident!("set_{}", fn_ident);
        sig.ident = setter_ident.clone();

        let value_argument: PatType = parse_quote!(__value: #ty);
        sig.inputs.push(FnArg::Typed(value_argument.clone()));

        // make `&self` `&mut self` instead.
        let mut_receiver: Receiver = parse_quote!(&mut self);
        if let Some(og) = sig.inputs.first_mut() {
            *og = FnArg::Receiver(mut_receiver)
        }

        // remove the return value.
        sig.output = ReturnType::Default;

        let value = &value_argument.pat;
        let method = quote! {
            #sig {
                use salsa::Setter;
                let data = #create_data_ident(self);
                data.#setter_ident(self).to(Some(#value));
            }
        };
        method.to_tokens(tokens);
    }
}

pub(crate) struct InputSetterWithDurability {
    pub(crate) signature: syn::Signature,
    pub(crate) return_type: syn::Type,
    pub(crate) create_data_ident: Ident,
}

impl ToTokens for InputSetterWithDurability {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &mut self.signature.clone();

        let ty = &self.return_type;
        let fn_ident = &sig.ident;
        let setter_ident = format_ident!("set_{}", fn_ident);

        let create_data_ident = &self.create_data_ident;

        sig.ident = format_ident!("set_{}_with_durability", fn_ident);

        let value_argument: PatType = parse_quote!(__value: #ty);
        sig.inputs.push(FnArg::Typed(value_argument.clone()));

        let durability_argument: PatType = parse_quote!(durability: salsa::Durability);
        sig.inputs.push(FnArg::Typed(durability_argument.clone()));

        // make `&self` `&mut self` instead.
        let mut_receiver: Receiver = parse_quote!(&mut self);
        if let Some(og) = sig.inputs.first_mut() {
            *og = FnArg::Receiver(mut_receiver)
        }

        // remove the return value.
        sig.output = ReturnType::Default;

        let value = &value_argument.pat;
        let durability = &durability_argument.pat;
        let method = quote! {
            #sig {
                use salsa::Setter;
                let data = #create_data_ident(self);
                data.#setter_ident(self)
                    .with_durability(#durability)
                    .to(Some(#value));
            }
        };
        method.to_tokens(tokens);
    }
}

pub(crate) enum SetterKind {
    Plain(InputSetter),
    WithDurability(InputSetterWithDurability),
}

impl ToTokens for SetterKind {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            SetterKind::Plain(input_setter) => input_setter.to_tokens(tokens),
            SetterKind::WithDurability(input_setter_with_durability) => {
                input_setter_with_durability.to_tokens(tokens)
            }
        }
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
pub(crate) struct Intern {
    pub(crate) signature: syn::Signature,
    pub(crate) pat_and_tys: Vec<PatType>,
    pub(crate) interned_struct_path: Path,
}

impl ToTokens for Intern {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &self.signature;

        let ty = self.pat_and_tys.to_vec();

        let interned_pat = ty.first().expect("at least one pat; this is a bug");
        let interned_pat = &interned_pat.pat;

        let wrapper_struct = self.interned_struct_path.to_token_stream();

        let method = quote! {
            #sig {
                #wrapper_struct::new(self, #interned_pat)
            }
        };

        method.to_tokens(tokens);
    }
}

pub(crate) struct Lookup {
    pub(crate) signature: syn::Signature,
    pub(crate) pat_and_tys: Vec<PatType>,
    pub(crate) return_ty: Type,
    pub(crate) interned_struct_path: Path,
}

impl Lookup {
    pub(crate) fn prepare_signature(&mut self) {
        let sig = &self.signature;

        let ident = format_ident!("lookup_{}", sig.ident);

        let ty = self.pat_and_tys.to_vec();

        let interned_key = &self.return_ty;

        let interned_pat = ty.first().expect("at least one pat; this is a bug");
        let interned_return_ty = &interned_pat.ty;

        self.signature = parse_quote!(
            fn #ident(&self, id: #interned_key) -> #interned_return_ty
        );
    }
}

impl ToTokens for Lookup {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let sig = &self.signature;

        let wrapper_struct = self.interned_struct_path.to_token_stream();
        let method = quote! {
            #sig {
                #wrapper_struct::ingredient(self).data(self.as_dyn_database(), id.as_id()).0.clone()
            }
        };

        method.to_tokens(tokens);
    }
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Queries {
    TrackedQuery(TrackedQuery),
    InputQuery(InputQuery),
    Intern(Intern),
    Transparent(Transparent),
}

impl ToTokens for Queries {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Queries::TrackedQuery(tracked_query) => tracked_query.to_tokens(tokens),
            Queries::InputQuery(input_query) => input_query.to_tokens(tokens),
            Queries::Transparent(transparent) => transparent.to_tokens(tokens),
            Queries::Intern(intern) => intern.to_tokens(tokens),
        }
    }
}
