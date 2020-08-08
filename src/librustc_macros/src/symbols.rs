use indexmap::IndexMap;
use proc_macro::TokenStream;
use proc_macro2::TokenTree;
use quote::quote;
use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, parse_macro_input, Ident, LitStr, Token};

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
        input.parse::<Token![,]>()?;

        Ok(Symbol { name, value })
    }
}

// Map from an optional keyword class to the list of keywords in it.
// FIXME: the indexmap crate thinks `has_std` is false when building `rustc_macros`,
// so we have to provide the hasher manually.
struct Keywords(IndexMap<Option<Ident>, Vec<Keyword>, RandomState>);

impl Parse for Keywords {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut classes = IndexMap::<_, Vec<_>, _>::with_hasher(Default::default());
        let mut current_class = None;
        while !input.is_empty() {
            if input.peek(Token![fn]) {
                input.parse::<TokenTree>()?;
                current_class = Some(input.parse::<Ident>()?);
                input.parse::<Token![:]>()?;
            } else {
                classes.entry(current_class.clone()).or_default().push(input.parse()?);
            }
        }
        Ok(Keywords(classes))
    }
}

struct Symbols(Vec<Symbol>);

impl Parse for Symbols {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut list = Vec::new();
        while !input.is_empty() {
            list.push(input.parse()?);
        }
        Ok(Symbols(list))
    }
}

struct Input {
    keywords: Keywords,
    symbols: Symbols,
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

        Ok(Input { keywords, symbols })
    }
}

pub fn symbols(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Input);

    let mut keyword_stream = quote! {};
    let mut symbols_stream = quote! {};
    let mut digits_stream = quote! {};
    let mut prefill_stream = quote! {};
    let mut keyword_class_stream = quote! {};
    let mut counter = 0u32;
    let mut keys = HashSet::<String>::new();
    let mut prev_key: Option<String> = None;
    let mut errors = Vec::<String>::new();

    let mut check_dup = |str: &str, errors: &mut Vec<String>| {
        if !keys.insert(str.to_string()) {
            errors.push(format!("Symbol `{}` is duplicated", str));
        }
    };

    let mut check_order = |str: &str, errors: &mut Vec<String>| {
        if let Some(ref prev_str) = prev_key {
            if str < prev_str {
                errors.push(format!("Symbol `{}` must precede `{}`", str, prev_str));
            }
        }
        prev_key = Some(str.to_string());
    };

    // Generate the listed keywords.
    for (class, keywords) in &input.keywords.0 {
        let mut class_stream = quote! {};
        for keyword in keywords {
            let name = &keyword.name;
            let value = &keyword.value;
            check_dup(&value.value(), &mut errors);
            prefill_stream.extend(quote! {
                #value,
            });
            keyword_stream.extend(quote! {
                #[allow(non_upper_case_globals)]
                pub const #name: Symbol = Symbol::new(#counter);
            });
            class_stream.extend(quote! {
                | kw::#name
            });
            counter += 1;
        }
        if let Some(class) = class {
            keyword_class_stream.extend(quote! {
                fn #class(self) -> bool {
                    match self {
                        #class_stream => true,
                        _ => false
                    }
                }
            });
        }
    }

    // Generate the listed symbols.
    for symbol in &input.symbols.0 {
        let name = &symbol.name;
        let value = match &symbol.value {
            Some(value) => value.value(),
            None => name.to_string(),
        };
        check_dup(&value, &mut errors);
        check_order(&name.to_string(), &mut errors);
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
        check_dup(&n, &mut errors);
        prefill_stream.extend(quote! {
            #n,
        });
        digits_stream.extend(quote! {
            Symbol::new(#counter),
        });
        counter += 1;
    }

    if !errors.is_empty() {
        for error in errors.into_iter() {
            eprintln!("error: {}", error)
        }
        panic!("errors in `Keywords` and/or `Symbols`");
    }

    let tt = TokenStream::from(quote! {
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

        impl Symbol {
            #keyword_class_stream
        }
    });

    // To see the generated code generated, uncomment this line, recompile, and
    // run the resulting output through `rustfmt`.
    //eprintln!("{}", tt);

    tt
}
