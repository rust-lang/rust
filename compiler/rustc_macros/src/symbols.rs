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
//! # The specific version number in CFG_RELEASE doesn't matter.
//! # The output is large.
//! CFG_RELEASE="0.0.0" cargo +nightly expand > /tmp/rustc_span.rs
//! ```

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, Lit, LitStr, Macro, Token, braced, bracketed};

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
    value: Value,
}

enum Value {
    SameAsName,
    String(LitStr),
    Env(LitStr, Macro),
    Unsupported(Expr),
}

impl Parse for Symbol {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        let colon_token: Option<Token![:]> = input.parse()?;
        let value = if colon_token.is_some() { input.parse()? } else { Value::SameAsName };

        Ok(Symbol { name, value })
    }
}

impl Parse for Value {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let expr: Expr = input.parse()?;
        match &expr {
            Expr::Lit(expr) => {
                if let Lit::Str(lit) = &expr.lit {
                    return Ok(Value::String(lit.clone()));
                }
            }
            Expr::Macro(expr) => {
                if expr.mac.path.is_ident("env")
                    && let Ok(lit) = expr.mac.parse_body()
                {
                    return Ok(Value::Env(lit, expr.mac.clone()));
                }
            }
            _ => {}
        }
        Ok(Value::Unsupported(expr))
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

pub(super) fn symbols(input: TokenStream) -> TokenStream {
    let (mut output, errors) = symbols_with_errors(input);

    // If we generated any errors, then report them as compiler_error!() macro calls.
    // This lets the errors point back to the most relevant span. It also allows us
    // to report as many errors as we can during a single run.
    output.extend(errors.into_iter().map(|e| e.to_compile_error()));

    output
}

struct Predefined {
    idx: u32,
    span_of_name: Span,
}

struct Duplicate {
    name: String,
    span_of_name: Span,
}

struct Entries {
    map: HashMap<String, Predefined>,
    prefill_stream: TokenStream,
}

impl Entries {
    fn with_capacity(capacity: usize) -> Self {
        Entries { map: HashMap::with_capacity(capacity), prefill_stream: TokenStream::new() }
    }

    fn try_insert(&mut self, span: Span, s: String) -> (u32, Option<Duplicate>) {
        let len = self.len();
        match self.map.entry(s) {
            Entry::Occupied(entry) => {
                let Predefined { idx, span_of_name } = *entry.get();
                (idx, Some(Duplicate { name: entry.key().clone(), span_of_name }))
            }
            Entry::Vacant(entry) => {
                let s = entry.key().as_str();
                self.prefill_stream.extend(quote! { #s, });
                entry.insert(Predefined { idx: len, span_of_name: span });
                (len, None)
            }
        }
    }

    fn insert(&mut self, span: Span, s: String, errors: &mut Errors) -> u32 {
        let (idx, duplicate) = self.try_insert(span, s);
        if let Some(Duplicate { name, span_of_name }) = duplicate {
            errors.error(span, format!("Symbol `{name}` is duplicated"));
            errors.error(span_of_name, "location of previous definition".to_string());
        }
        idx
    }

    fn len(&self) -> u32 {
        u32::try_from(self.map.len()).expect("way too many symbols")
    }
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
    let mut entries = Entries::with_capacity(input.keywords.len() + input.symbols.len() + 10);

    // Generate the listed keywords.
    for keyword in input.keywords.iter() {
        let name = &keyword.name;
        let value_string = keyword.value.value();
        let idx = entries.insert(keyword.name.span(), value_string, &mut errors);
        keyword_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    // Generate the listed symbols.
    for symbol in input.symbols.iter() {
        let name = &symbol.name;

        let value = match &symbol.value {
            Value::SameAsName => name.to_string(),
            Value::String(lit) => lit.value(),
            Value::Env(..) => continue, // in another loop below
            Value::Unsupported(expr) => {
                errors.list.push(syn::Error::new_spanned(
                    expr,
                    concat!(
                        "unsupported expression for symbol value; implement support for this in ",
                        file!(),
                    ),
                ));
                continue;
            }
        };
        let idx = entries.insert(symbol.name.span(), value, &mut errors);
        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    // Generate symbols for the strings "0", "1", ..., "9".
    for n in 0..10 {
        entries.insert(Span::call_site(), n.to_string(), &mut errors);
    }

    // Symbols whose value comes from an environment variable. It's allowed for
    // these to have the same value as another symbol.
    for symbol in &input.symbols {
        let (env_var, expr) = match &symbol.value {
            Value::Env(lit, expr) => (lit, expr),
            Value::SameAsName | Value::String(_) | Value::Unsupported(_) => continue,
        };

        if !proc_macro::is_available() {
            errors.error(
                Span::call_site(),
                "proc_macro::tracked_env is not available in unit test".to_owned(),
            );
            break;
        }

        let value = match proc_macro::tracked_env::var(env_var.value()) {
            Ok(value) => value,
            Err(err) => {
                errors.list.push(syn::Error::new_spanned(expr, err));
                continue;
            }
        };

        let name = &symbol.name;
        let (idx, _) = entries.try_insert(name.span(), value);
        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    let symbol_digits_base = entries.map["0"].idx;
    let predefined_symbols_count = entries.len();
    let prefill_stream = entries.prefill_stream;
    let output = quote! {
        const SYMBOL_DIGITS_BASE: u32 = #symbol_digits_base;

        /// The number of predefined symbols; this is the first index for
        /// extra pre-interned symbols in an Interner created via
        /// [`Interner::with_extra_symbols`].
        pub const PREDEFINED_SYMBOLS_COUNT: u32 = #predefined_symbols_count;

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        mod kw_generated {
            use super::Symbol;
            #keyword_stream
        }

        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        pub mod sym_generated {
            use super::Symbol;
            #symbols_stream
        }

        impl Interner {
            /// Creates an `Interner` with the predefined symbols from the `symbols!` macro and
            /// any extra symbols provided by external drivers such as Clippy
            pub(crate) fn new(driver_symbols: Option<&[&'static str]>) -> Self {
                Interner::prefill(
                    driver_symbols.unwrap_or(&[#prefill_stream])
                )
            }
        }

        /// Allows drivers to define extra preinterned symbols, the expanded `PREINTERNED_SYMBOLS`
        /// is to be provided to `rustc_interface::Config`
        #[macro_export]
        macro_rules! extra_symbols {
            ($(#[macro_export] $further_symbols:ident;)? Symbols { $($tt:tt)* }) => {
                rustc_macros::extra_symbols_impl! {
                    $($further_symbols)? [#prefill_stream] $($tt)*
                }
            };
        }
    };

    (output, errors.list)
}

#[derive(Default)]
struct ExtraSymbols {
    macro_ident: Option<Ident>,
    predefined: Punctuated<LitStr, Token![,]>,
    extra_symbols: Punctuated<Symbol, Token![,]>,
}

impl Parse for ExtraSymbols {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let macro_ident: Option<Ident> = input.parse()?;

        let content;
        bracketed!(content in input);
        let predefined = Punctuated::parse_terminated(&content)?;

        let extra_symbols = Punctuated::parse_terminated(&input)?;

        Ok(ExtraSymbols { macro_ident, predefined, extra_symbols })
    }
}

pub(super) fn extra_symbols(input: TokenStream) -> TokenStream {
    let mut errors = Errors::default();

    let input: ExtraSymbols = match syn::parse2(input) {
        Ok(input) => input,
        Err(e) => {
            errors.list.push(e);
            Default::default()
        }
    };

    let mut symbol_stream = TokenStream::new();
    let mut duplicate_symbols = TokenStream::new();

    let mut entries = Entries::with_capacity(input.predefined.len() + input.extra_symbols.len());
    for lit in &input.predefined {
        entries.insert(lit.span(), lit.value(), &mut errors);
    }

    for symbol in input.extra_symbols {
        let value = match symbol.value {
            Value::SameAsName => symbol.name.to_string(),
            Value::String(lit) => lit.value(),
            _ => {
                errors.error(
                    symbol.name.span(),
                    "unsupported expression for extra symbol value".to_string(),
                );
                continue;
            }
        };

        let name = &symbol.name;
        let (idx, duplicate) = entries.try_insert(name.span(), value);
        if duplicate.is_some() {
            duplicate_symbols.extend(quote! { #name, });
        }
        symbol_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    let prefill_stream = entries.prefill_stream;

    let further_symbols = if let Some(macro_ident) = input.macro_ident {
        quote! {
            #[macro_export]
            macro_rules! #macro_ident {
                ($(#[macro_export] $further_symbols:ident;)? Symbols { $($tt:tt)* }) => {
                    rustc_macros::extra_symbols_impl! {
                        $($further_symbols)? [#prefill_stream] $($tt)*
                    }
                };
            }
        }
    } else {
        TokenStream::new()
    };

    let mut output = quote! {
        /// To be supplied to `rustc_interface::Config`
        pub const PREINTERNED_SYMBOLS: &[&str] = &[
            #prefill_stream
        ];

        pub const DUPLICATE_SYMBOLS: &[Symbol] = &[
            #duplicate_symbols
        ];

        #symbol_stream

        #further_symbols
    };

    output.extend(errors.list.into_iter().map(|e| e.into_compile_error()));

    output
}
