//! Proc macro which builds the Symbol table
//!
//! # Debugging
//!
//! Since this proc-macro does some non-trivial work, debugging it is important.
//! This proc-macro can be invoked as an ordinary unit test, like so:
//!
//! ```bash
//! cd compiler/rustc_macros
//! cargo test symbols::test_symbols -- --nocapture
//! ```
//!
//! This unit test finds the `symbols!` invocation in `compiler/rustc_span/src/symbol.rs`
//! and runs it. It verifies that the output token stream can be parsed as valid module
//! items and that no errors were produced.
//!
//! You can also view the generated code by using `cargo expand`:
//!
//! ```bash
//! cargo install cargo-expand          # this is necessary only once
//! cd compiler/rustc_span
//! cargo expand > /tmp/rustc_span.rs   # it's a big file
//! ```

use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::HashMap;
use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, punctuated::Punctuated, Ident, LitStr, Token};

#[cfg(test)]
mod tests;

mod kw {
    syn::custom_keyword!(Keywords);
    syn::custom_keyword!(Symbols);
}

struct Keyword {
    name: Ident,
    value: LitStr,
}

impl Parse for Keyword {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let value = input.parse()?;

        Ok(Keyword { name, value })
    }
}

struct Symbol {
    name: Ident,
    value: Option<LitStr>,
}

impl Parse for Symbol {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        let value = match input.parse::<Token![:]>() {
            Ok(_) => Some(input.parse()?),
            Err(_) => None,
        };

        Ok(Symbol { name, value })
    }
}

struct Input {
    keywords: Punctuated<Keyword, Token![,]>,
    symbols: Punctuated<Symbol, Token![,]>,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<kw::Keywords>()?;
        let content;
        braced!(content in input);
        let keywords = Punctuated::parse_terminated(&content)?;

        input.parse::<kw::Symbols>()?;
        let content;
        braced!(content in input);
        let symbols = Punctuated::parse_terminated(&content)?;

        Ok(Input { keywords, symbols })
    }
}

#[derive(Default)]
struct Errors {
    list: Vec<syn::Error>,
}

impl Errors {
    fn error(&mut self, span: Span, message: String) {
        self.list.push(syn::Error::new(span, message));
    }
}

pub fn symbols(input: TokenStream) -> TokenStream {
    let (mut output, errors) = symbols_with_errors(input);

    // If we generated any errors, then report them as compiler_error!() macro calls.
    // This lets the errors point back to the most relevant span. It also allows us
    // to report as many errors as we can during a single run.
    output.extend(errors.into_iter().map(|e| e.to_compile_error()));

    output
}

fn symbols_with_errors(input: TokenStream) -> (TokenStream, Vec<syn::Error>) {
    let mut errors = Errors::default();

    let input: Input = match syn::parse2(input) {
        Ok(input) => input,
        Err(e) => {
            // This allows us to display errors at the proper span, while minimizing
            // unrelated errors caused by bailing out (and not generating code).
            errors.list.push(e);
            Input { keywords: Default::default(), symbols: Default::default() }
        }
    };

    let mut keyword_stream = quote! {};
    let mut symbols_stream = quote! {};
    let mut digits_stream = quote! {};
    let mut prefill_stream = quote! {};
    let mut counter = 0u32;
    let mut keys =
        HashMap::<String, Span>::with_capacity(input.keywords.len() + input.symbols.len() + 10);
    let mut prev_key: Option<(Span, String)> = None;

    let mut check_dup = |span: Span, str: &str, errors: &mut Errors| {
        if let Some(prev_span) = keys.get(str) {
            errors.error(span, format!("Symbol `{}` is duplicated", str));
            errors.error(*prev_span, format!("location of previous definition"));
        } else {
            keys.insert(str.to_string(), span);
        }
    };

    let mut check_order = |span: Span, str: &str, errors: &mut Errors| {
        if let Some((prev_span, ref prev_str)) = prev_key {
            if str < prev_str {
                errors.error(span, format!("Symbol `{}` must precede `{}`", str, prev_str));
                errors.error(prev_span, format!("location of previous symbol `{}`", prev_str));
            }
        }
        prev_key = Some((span, str.to_string()));
    };

    // Generate the listed keywords.
    for keyword in input.keywords.iter() {
        let name = &keyword.name;
        let value = &keyword.value;
        let value_string = value.value();
        check_dup(keyword.name.span(), &value_string, &mut errors);
        prefill_stream.extend(quote! {
            #value,
        });
        keyword_stream.extend(quote! {
            #[allow(non_upper_case_globals)]
            pub const #name: Symbol = Symbol::new(#counter);
        });
        counter += 1;
    }

    // Generate the listed symbols.
    for symbol in input.symbols.iter() {
        let name = &symbol.name;
        let value = match &symbol.value {
            Some(value) => value.value(),
            None => name.to_string(),
        };
        check_dup(symbol.name.span(), &value, &mut errors);
        check_order(symbol.name.span(), &name.to_string(), &mut errors);

        prefill_stream.extend(quote! {
            #value,
        });
        symbols_stream.extend(quote! {
            #[allow(rustc::default_hash_types)]
            #[allow(non_upper_case_globals)]
            pub const #name: Symbol = Symbol::new(#counter);
        });
        counter += 1;
    }

    // Generate symbols for the strings "0", "1", ..., "9".
    for n in 0..10 {
        let n = n.to_string();
        check_dup(Span::call_site(), &n, &mut errors);
        prefill_stream.extend(quote! {
            #n,
        });
        digits_stream.extend(quote! {
            Symbol::new(#counter),
        });
        counter += 1;
    }

    let output = quote! {
        macro_rules! keywords {
            () => {
                #keyword_stream
            }
        }

        macro_rules! define_symbols {
            () => {
                #symbols_stream

                #[allow(non_upper_case_globals)]
                pub const digits_array: &[Symbol; 10] = &[
                    #digits_stream
                ];
            }
        }

        impl Interner {
            pub fn fresh() -> Self {
                Interner::prefill(&[
                    #prefill_stream
                ])
            }
        }
    };

    (output, errors.list)

    // To see the generated code, use the "cargo expand" command.
    // Do this once to install:
    //      cargo install cargo-expand
    //
    // Then, cd to rustc_span and run:
    //      cargo expand > /tmp/rustc_span_expanded.rs
    //
    // and read that file.
}
