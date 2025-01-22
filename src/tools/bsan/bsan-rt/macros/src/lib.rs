use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident, LitInt, Token};
use syn::parse::{Parse, ParseStream};

struct StaticInitInput {
    prefix: Ident,
    value: LitInt,
}

impl Parse for StaticInitInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let prefix: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let value: LitInt = input.parse()?;
        Ok(Self { prefix, value })
    }
}

#[proc_macro]
pub fn init_statics(input: TokenStream) -> TokenStream {
    let StaticInitInput { prefix, value } = parse_macro_input!(input as StaticInitInput);

    let sig_ident = Ident::new(&format!("{}_SIG", prefix), prefix.span());
    let l1_ident = Ident::new(&format!("{}_L1", prefix), prefix.span());
    let l2_ident = Ident::new(&format!("{}_L2", prefix), prefix.span());

    let value_expr = value.base10_parse::<usize>().unwrap();
    let l1_expr = (value_expr / 2).pow(2);
    let l2_expr = ((value_expr + 1) / 2).pow(2);

    let expanded = quote! {
        pub static #sig_ident: usize = #value_expr;
        pub static #l1_ident: usize = #l1_expr;
        pub static #l2_ident: usize = #l2_expr;
    };

    TokenStream::from(expanded)
}