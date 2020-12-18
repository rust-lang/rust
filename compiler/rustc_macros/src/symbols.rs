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

    let mut keys =
        HashMap::<String, Span>::with_capacity(input.keywords.len() + input.symbols.len() + 10);
    let mut prev_key: Option<(Span, String)> = None;

    let mut check_dup = |span: Span, str: &str, errors: &mut Errors| {
        if let Some(prev_span) = keys.get(str) {
            errors.error(span, format!("Symbol `{}` is duplicated", str));
            errors.error(*prev_span, format!("location of previous definition"));
            Err(())
        } else {
            keys.insert(str.to_string(), span);
            Ok(())
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

    let mut symbol_strings: Vec<String> = Vec::new();

    // Generate the listed keywords.
    let mut keyword_stream = quote! {};
    for keyword in input.keywords.iter() {
        let name = &keyword.name;
        let value = &keyword.value;
        let value_string = value.value();
        let symbol_index = symbol_strings.len() as u32;
        if check_dup(keyword.name.span(), &value_string, &mut errors).is_ok() {
            // Only add an entry to `symbol_strings` if it is not a duplicate.
            // If it is a duplicate, then compilation will fail. However, we still
            // want to avoid panicking, if a duplicate is detected.
            symbol_strings.push(value_string);
        }
        keyword_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#symbol_index);
        });
    }

    // Generate symbols for the strings "0", "1", ..., "9".
    let digits_base = symbol_strings.len() as u32;
    for n in 0..10 {
        let n_string = n.to_string();
        if check_dup(Span::call_site(), &n_string, &mut errors).is_ok() {
            symbol_strings.push(n_string);
        }
    }

    // Generate the listed symbols.
    let mut symbols_stream = quote! {};
    for symbol in input.symbols.iter() {
        let name = &symbol.name;
        let name_string = symbol.name.to_string();
        check_order(symbol.name.span(), &name_string, &mut errors);
        let value = match &symbol.value {
            Some(value) => value.value(),
            None => name_string,
        };

        let symbol_index = symbol_strings.len() as u32;
        if check_dup(symbol.name.span(), &value, &mut errors).is_ok() {
            // Only add an entry to `symbol_strings` if it is not a duplicate.
            // If it is a duplicate, then compilation will fail. However, we still
            // want to avoid panicking, if a duplicate is detected.
            symbol_strings.push(value);
        }

        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#symbol_index);
        });
    }

    // We have finished collecting symbol strings.
    let static_symbols_len = symbol_strings.len();
    let dynamic_symbol_base = symbol_strings.len() as u32;
    let symbol_strings = symbol_strings; // no more mutation

    // Build the body of STATIC_SYMBOLS.
    let symbol_strings_tokens: TokenStream = symbol_strings.iter().map(|s| quote!(#s,)).collect();

    // Build the PHF map. This translates from strings to Symbol values.
    let mut phf_map = phf_codegen::Map::<&str>::new();
    for (symbol_index, symbol) in symbol_strings.iter().enumerate() {
        phf_map.entry(symbol, format!("Symbol::new({})", symbol_index as u32).as_str());
    }
    let phf_map_built = phf_map.build();
    let phf_map_text = phf_map_built.to_string();
    let phf_map_expr = syn::parse_str::<syn::Expr>(&phf_map_text).unwrap();

    let output = quote! {
        const SYMBOL_DIGITS_BASE: u32 = #digits_base;

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        mod kw_generated {
            use super::Symbol;
            #keyword_stream
        }

        #[allow(rustc::default_hash_types)]
        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        pub mod sym_generated {
            use super::Symbol;
            #symbols_stream
        }

        const DYNAMIC_SYMBOL_BASE: u32 = #dynamic_symbol_base;

        static STATIC_SYMBOLS: [&str; #static_symbols_len as usize] = [
            #symbol_strings_tokens
        ];

        static STATIC_SYMBOLS_PHF: ::phf::Map<&'static str, Symbol> = #phf_map_expr;
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
