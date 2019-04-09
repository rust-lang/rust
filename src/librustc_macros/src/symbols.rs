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
    syn::custom_keyword!(Other);
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

struct Symbol(Ident);

impl Parse for Symbol {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ident: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        Ok(Symbol(ident))
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

        input.parse::<kw::Other>()?;
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
    let mut prefill_stream = quote! {};
    let mut from_str_stream = quote! {};
    let mut counter = 0u32;
    let mut keys = HashSet::<String>::new();

    let mut check_dup = |str: &str| {
        if !keys.insert(str.to_string()) {
            panic!("Symbol `{}` is duplicated", str);
        }
    };

    for keyword in &input.keywords.0 {
        let name = &keyword.name;
        let value = &keyword.value;
        check_dup(&value.value());
        prefill_stream.extend(quote! {
            #value,
        });
        keyword_stream.extend(quote! {
            pub const #name: Keyword = Keyword {
                ident: Ident::with_empty_ctxt(super::Symbol::new(#counter))
            };
        });
        from_str_stream.extend(quote! {
            #value => Ok(#name),
        });
        counter += 1;
    }

    for symbol in &input.symbols.0 {
        let value = &symbol.0;
        let value_str = value.to_string();
        check_dup(&value_str);
        prefill_stream.extend(quote! {
            #value_str,
        });
        symbols_stream.extend(quote! {
            pub const #value: Symbol = Symbol::new(#counter);
        });
        counter += 1;
    }

    TokenStream::from(quote! {
        macro_rules! keywords {
            () => {
                #keyword_stream

                impl std::str::FromStr for Keyword {
                    type Err = ();

                    fn from_str(s: &str) -> Result<Self, ()> {
                        match s {
                            #from_str_stream
                            _ => Err(()),
                        }
                    }
                }
            }
        }

        macro_rules! symbols {
            () => {
                #symbols_stream
            }
        }

        impl Interner {
            pub fn fresh() -> Self {
                Interner::prefill(&[
                    #prefill_stream
                ])
            }
        }
    })
}
