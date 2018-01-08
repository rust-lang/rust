//! Implementation of the `#[assert_instr]` macro
//!
//! This macro is used when testing the `stdsimd` crate and is used to generate
//! test cases to assert that functions do indeed contain the instructions that
//! we're expecting them to contain.
//!
//! The procedural macro here is relatively simple, it simply appends a
//! `#[test]` function to the original token stream which asserts that the
//! function itself contains the relevant instruction.

#![feature(proc_macro)]

extern crate proc_macro2;
extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro2::TokenStream;

#[proc_macro_attribute]
pub fn assert_instr(
    attr: proc_macro::TokenStream, item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let invoc = syn::parse::<Invoc>(attr)
        .expect("expected #[assert_instr(instr, a = b, ...)]");
    let item =
        syn::parse::<syn::Item>(item).expect("must be attached to an item");
    let func = match item {
        syn::Item::Fn(ref f) => f,
        _ => panic!("must be attached to a function"),
    };

    let instr = &invoc.instr;
    let maybe_ignore = if cfg!(optimized) {
        TokenStream::empty()
    } else {
        (quote! { #[ignore] }).into()
    };
    let name = &func.ident;
    let assert_name = syn::Ident::from(
        &format!("assert_{}_{}", name.as_ref(), instr.as_ref())[..],
    );
    let shim_name = syn::Ident::from(format!("{}_shim", name.as_ref()));
    let (to_test, test_name) = if invoc.args.len() == 0 {
        (TokenStream::empty(), &func.ident)
    } else {
        let mut inputs = Vec::new();
        let mut input_vals = Vec::new();
        let ret = &func.decl.output;
        for arg in func.decl.inputs.iter() {
            let capture = match *arg {
                syn::FnArg::Captured(ref c) => c,
                _ => panic!("arguments must not have patterns"),
            };
            let ident = match capture.pat {
                syn::Pat::Ident(ref i) => &i.ident,
                _ => panic!("must have bare arguments"),
            };
            match invoc.args.iter().find(|a| a.0 == ident.as_ref()) {
                Some(&(_, ref tts)) => {
                    input_vals.push(quote! { #tts });
                }
                None => {
                    inputs.push(capture);
                    input_vals.push(quote! { #ident });
                }
            };
        }

        let attrs = func.attrs
            .iter()
            .filter(|attr| {
                attr.path
                    .segments
                    .first()
                    .unwrap()
                    .value()
                    .ident
                    .as_ref()
                    .starts_with("target")
            })
            .collect::<Vec<_>>();
        let attrs = Append(&attrs);
        (
            quote! {
                #attrs
                unsafe fn #shim_name(#(#inputs),*) #ret {
                    #name(#(#input_vals),*)
                }
            }.into(),
            &shim_name,
        )
    };

    let tts: TokenStream = quote_spanned! {
        proc_macro2::Span::call_site() =>
        #[test]
        #[allow(non_snake_case)]
        #maybe_ignore
        fn #assert_name() {
            #to_test

            ::stdsimd_test::assert(#test_name as usize,
                                   stringify!(#test_name),
                                   stringify!(#instr));
        }
    }.into();
    // why? necessary now to get tests to work?
    let tts: TokenStream = tts.to_string().parse().unwrap();

    let tts: TokenStream = quote! {
        #item
        #tts
    }.into();
    tts.into()
}

struct Invoc {
    instr: syn::Ident,
    args: Vec<(syn::Ident, syn::Expr)>,
}

impl syn::synom::Synom for Invoc {
    named!(parse -> Self, map!(parens!(do_parse!(
        instr: syn!(syn::Ident) >>
        args: many0!(do_parse!(
            syn!(syn::token::Comma) >>
            name: syn!(syn::Ident) >>
            syn!(syn::token::Eq) >>
            expr: syn!(syn::Expr) >>
            (name, expr)
        )) >>
        (Invoc {
            instr,
            args,
        })
    )), |p| p.1));
}

struct Append<T>(T);

impl<T> quote::ToTokens for Append<T>
where
    T: Clone + IntoIterator,
    T::Item: quote::ToTokens,
{
    fn to_tokens(&self, tokens: &mut quote::Tokens) {
        for item in self.0.clone() {
            item.to_tokens(tokens);
        }
    }
}
