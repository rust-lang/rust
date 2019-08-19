use proc_macro::TokenStream;
use syn::{
    Token, Ident, LitStr,
    braced, parse_macro_input,
};
use syn::parse::{Result, Parse, ParseStream};
use syn;
use std::collections::HashSet;
use quote::quote;

#[allow(non_camel_case_types)]
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
        input.parse::<Token![,]>()?;

        Ok(Keyword {
            name,
            value,
        })
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
        input.parse::<Token![,]>()?;

        Ok(Symbol {
            name,
            value,
        })
    }
}

/// A type used to greedily parse another type until the input is empty.
struct List<T>(Vec<T>);

impl<T: Parse> Parse for List<T> {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut list = Vec::new();
        while !input.is_empty() {
            list.push(input.parse()?);
        }
        Ok(List(list))
    }
}

struct Input {
    keywords: List<Keyword>,
    symbols: List<Symbol>,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<kw::Keywords>()?;
        let content;
        braced!(content in input);
        let keywords = content.parse()?;

        input.parse::<kw::Symbols>()?;
        let content;
        braced!(content in input);
        let symbols = content.parse()?;

        Ok(Input {
            keywords,
            symbols,
        })
    }
}

pub fn symbols(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Input);

    let mut keyword_stream = quote! {};
    let mut symbols_stream = quote! {};
    let mut digits_stream = quote! {};
    let mut prefill_stream = quote! {};
    let mut counter = 0u32;
    let mut keys = HashSet::<String>::new();

    let mut check_dup = |str: &str| {
        if !keys.insert(str.to_string()) {
            panic!("Symbol `{}` is duplicated", str);
        }
    };

    // Generate the listed keywords.
    for keyword in &input.keywords.0 {
        let name = &keyword.name;
        let value = &keyword.value;
        check_dup(&value.value());
        prefill_stream.extend(quote! {
            #value,
        });
        keyword_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#counter);
        });
        counter += 1;
    }

    // Generate the listed symbols.
    for symbol in &input.symbols.0 {
        let name = &symbol.name;
        let value = match &symbol.value {
            Some(value) => value.value(),
            None => name.to_string(),
        };
        check_dup(&value);
        prefill_stream.extend(quote! {
            #value,
        });
        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#counter);
        });
        counter += 1;
    }

    // Generate symbols for the strings "0", "1", ..., "9".
    for n in 0..10 {
        let n = n.to_string();
        check_dup(&n);
        prefill_stream.extend(quote! {
            #n,
        });
        digits_stream.extend(quote! {
            Symbol::new(#counter),
        });
        counter += 1;
    }

    let tt = TokenStream::from(quote! {
        macro_rules! keywords {
            () => {
                #keyword_stream
            }
        }

        macro_rules! symbols {
            () => {
                #symbols_stream

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
    });

    // To see the generated code generated, uncomment this line, recompile, and
    // run the resulting output through `rustfmt`.
    //eprintln!("{}", tt);

    tt
}
