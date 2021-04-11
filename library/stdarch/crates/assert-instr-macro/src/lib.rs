//! Implementation of the `#[assert_instr]` macro
//!
//! This macro is used when testing the `stdarch` crate and is used to generate
//! test cases to assert that functions do indeed contain the instructions that
//! we're expecting them to contain.
//!
//! The procedural macro here is relatively simple, it simply appends a
//! `#[test]` function to the original token stream which asserts that the
//! function itself contains the relevant instruction.
#![deny(rust_2018_idioms)]

#[macro_use]
extern crate quote;

use proc_macro2::TokenStream;
use quote::ToTokens;

#[proc_macro_attribute]
pub fn assert_instr(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let invoc = match syn::parse::<Invoc>(attr) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };
    let item = match syn::parse::<syn::Item>(item) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };
    let func = match item {
        syn::Item::Fn(ref f) => f,
        _ => panic!("must be attached to a function"),
    };

    let instr = &invoc.instr;
    let name = &func.sig.ident;

    // Disable assert_instr for x86 targets compiled with avx enabled, which
    // causes LLVM to generate different intrinsics that the ones we are
    // testing for.
    let disable_assert_instr = std::env::var("STDARCH_DISABLE_ASSERT_INSTR").is_ok();

    // If instruction tests are disabled avoid emitting this shim at all, just
    // return the original item without our attribute.
    if !cfg!(optimized) || disable_assert_instr {
        return (quote! { #item }).into();
    }

    let instr_str = instr
        .replace('.', "_")
        .replace('/', "_")
        .replace(':', "_")
        .replace(char::is_whitespace, "");
    let assert_name = syn::Ident::new(&format!("assert_{}_{}", name, instr_str), name.span());
    // These name has to be unique enough for us to find it in the disassembly later on:
    let shim_name = syn::Ident::new(
        &format!("stdarch_test_shim_{}_{}", name, instr_str),
        name.span(),
    );
    let mut inputs = Vec::new();
    let mut input_vals = Vec::new();
    let mut const_vals = Vec::new();
    let ret = &func.sig.output;
    for arg in func.sig.inputs.iter() {
        let capture = match *arg {
            syn::FnArg::Typed(ref c) => c,
            ref v => panic!(
                "arguments must not have patterns: `{:?}`",
                v.clone().into_token_stream()
            ),
        };
        let ident = match *capture.pat {
            syn::Pat::Ident(ref i) => &i.ident,
            _ => panic!("must have bare arguments"),
        };
        if let Some(&(_, ref tokens)) = invoc.args.iter().find(|a| *ident == a.0) {
            input_vals.push(quote! { #tokens });
        } else {
            inputs.push(capture);
            input_vals.push(quote! { #ident });
        }
    }
    for arg in func.sig.generics.params.iter() {
        let c = match *arg {
            syn::GenericParam::Const(ref c) => c,
            ref v => panic!(
                "only const generics are allowed: `{:?}`",
                v.clone().into_token_stream()
            ),
        };
        if let Some(&(_, ref tokens)) = invoc.args.iter().find(|a| c.ident == a.0) {
            const_vals.push(quote! { #tokens });
        } else {
            panic!("const generics must have a value for tests");
        }
    }

    let attrs = func
        .attrs
        .iter()
        .filter(|attr| {
            attr.path
                .segments
                .first()
                .expect("attr.path.segments.first() failed")
                .ident
                .to_string()
                .starts_with("target")
        })
        .collect::<Vec<_>>();
    let attrs = Append(&attrs);

    // Use an ABI on Windows that passes SIMD values in registers, like what
    // happens on Unix (I think?) by default.
    let abi = if cfg!(windows) {
        syn::LitStr::new("vectorcall", proc_macro2::Span::call_site())
    } else {
        syn::LitStr::new("C", proc_macro2::Span::call_site())
    };
    let shim_name_str = format!("{}{}", shim_name, assert_name);
    let to_test = quote! {
        #attrs
        #[no_mangle]
        #[inline(never)]
        pub unsafe extern #abi fn #shim_name(#(#inputs),*) #ret {
            // The compiler in optimized mode by default runs a pass called
            // "mergefunc" where it'll merge functions that look identical.
            // Turns out some intrinsics produce identical code and they're
            // folded together, meaning that one just jumps to another. This
            // messes up our inspection of the disassembly of this function and
            // we're not a huge fan of that.
            //
            // To thwart this pass and prevent functions from being merged we
            // generate some code that's hopefully very tight in terms of
            // codegen but is otherwise unique to prevent code from being
            // folded.
            //
            // This is avoided on Wasm32 right now since these functions aren't
            // inlined which breaks our tests since each intrinsic looks like it
            // calls functions. Turns out functions aren't similar enough to get
            // merged on wasm32 anyway. This bug is tracked at
            // rust-lang/rust#74320.
            #[cfg(not(target_arch = "wasm32"))]
            ::stdarch_test::_DONT_DEDUP.store(
                std::mem::transmute(#shim_name_str.as_bytes().as_ptr()),
                std::sync::atomic::Ordering::Relaxed,
            );
            #name::<#(#const_vals),*>(#(#input_vals),*)
        }
    };

    let tokens: TokenStream = quote! {
        #[test]
        #[allow(non_snake_case)]
        fn #assert_name() {
            #to_test

            ::stdarch_test::assert(#shim_name as usize,
                                   stringify!(#shim_name),
                                   #instr);
        }
    };

    let tokens: TokenStream = quote! {
        #item
        #tokens
    };
    tokens.into()
}

struct Invoc {
    instr: String,
    args: Vec<(syn::Ident, syn::Expr)>,
}

impl syn::parse::Parse for Invoc {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        use syn::{ext::IdentExt, Token};

        let mut instr = String::new();
        while !input.is_empty() {
            if input.parse::<Token![,]>().is_ok() {
                break;
            }
            if let Ok(ident) = syn::Ident::parse_any(input) {
                instr.push_str(&ident.to_string());
                continue;
            }
            if input.parse::<Token![.]>().is_ok() {
                instr.push('.');
                continue;
            }
            if let Ok(s) = input.parse::<syn::LitStr>() {
                instr.push_str(&s.value());
                continue;
            }
            println!("{:?}", input.cursor().token_stream());
            return Err(input.error("expected an instruction"));
        }
        if instr.is_empty() {
            return Err(input.error("expected an instruction before comma"));
        }
        let mut args = Vec::new();
        while !input.is_empty() {
            let name = input.parse::<syn::Ident>()?;
            input.parse::<Token![=]>()?;
            let expr = input.parse::<syn::Expr>()?;
            args.push((name, expr));

            if input.parse::<Token![,]>().is_err() {
                if !input.is_empty() {
                    return Err(input.error("extra tokens at end"));
                }
                break;
            }
        }
        Ok(Self { instr, args })
    }
}

struct Append<T>(T);

impl<T> quote::ToTokens for Append<T>
where
    T: Clone + IntoIterator,
    T::Item: quote::ToTokens,
{
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for item in self.0.clone() {
            item.to_tokens(tokens);
        }
    }
}
