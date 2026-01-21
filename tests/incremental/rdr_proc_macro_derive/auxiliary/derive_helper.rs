extern crate proc_macro;
use proc_macro::TokenStream;

#[cfg(rpass1)]
fn format_impl_body() -> &'static str {
    "42"
}

#[cfg(rpass2)]
fn format_impl_body() -> &'static str {
    "21 + 21"
}

#[cfg(rpass3)]
fn format_impl_body() -> &'static str {
    "40 + 2"
}

#[cfg(any(rpass2, rpass3))]
fn _unused_private_helper() -> u32 {
    999
}

#[cfg(rpass3)]
struct _PrivateHelperStruct {
    _field: u32,
}

#[proc_macro_derive(RdrTestDerive)]
pub fn derive_rdr_test(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    let struct_name = input_str
        .split_whitespace()
        .skip_while(|s| *s == "pub" || *s == "struct")
        .next()
        .map(|s| s.trim_end_matches(';'))
        .unwrap_or("Unknown");

    let body = format_impl_body();

    let output = format!(
        "impl {struct_name} {{ pub fn derived_value() -> u32 {{ {body} }} }}"
    );

    output.parse().unwrap()
}
