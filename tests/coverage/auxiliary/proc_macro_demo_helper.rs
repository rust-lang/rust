//@ edition: 2021
// (The proc-macro crate doesn't need to be instrumented.)
//@ compile-flags: -Cinstrument-coverage=off

use proc_macro::TokenStream;

/// Returns the name of the type that a derive macro was applied to.
fn type_name(input: TokenStream) -> String {
    let mut tokens = input.to_string();
    let name_start = tokens.find("struct").unwrap() + "struct".len();
    tokens.drain(..name_start);
    tokens.split_whitespace().next().unwrap().trim_end_matches(';').to_string()
}

/// Derive macro that generates methods entirely from macro-supplied tokens.
/// Parsing from a string gives all tokens call-site spans, as `quote!` does.
#[proc_macro_derive(Demo)]
pub fn derive_demo(input: TokenStream) -> TokenStream {
    let name = type_name(input);
    format!(
        "impl {name} {{
            fn demo(&self, x: u32) -> u32 {{
                if x > 10 {{ x * 2 }} else {{ x + 1 }}
            }}
            fn never_called(&self) -> u32 {{
                999
            }}
        }}"
    )
    .parse()
    .unwrap()
}

/// Attribute macro that discards the annotated function and replaces it with
/// a generated function of the same name, whose body consists entirely of
/// macro-supplied tokens with call-site spans.
#[proc_macro_attribute]
pub fn generate_body(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = item.to_string();
    let name_start = item.find("fn").unwrap() + "fn".len();
    let name = item[name_start..].split(['(', ' ']).find(|s| !s.is_empty()).unwrap();
    format!(
        "fn {name}(x: u32) -> u32 {{
            if x > 10 {{ x * 2 }} else {{ x + 1 }}
        }}"
    )
    .parse()
    .unwrap()
}
