//! Exports a few trivial procedural macros for testing.

use proc_macro::TokenStream;

#[proc_macro]
pub fn fn_like_noop(args: TokenStream) -> TokenStream {
    args
}

#[proc_macro]
pub fn fn_like_panic(args: TokenStream) -> TokenStream {
    panic!("fn_like_panic!({})", args);
}

#[proc_macro]
pub fn fn_like_error(args: TokenStream) -> TokenStream {
    format!("compile_error!(\"fn_like_error!({})\");", args).parse().unwrap()
}

#[proc_macro_attribute]
pub fn attr_noop(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn attr_panic(args: TokenStream, item: TokenStream) -> TokenStream {
    panic!("#[attr_panic {}] {}", args, item);
}

#[proc_macro_attribute]
pub fn attr_error(args: TokenStream, item: TokenStream) -> TokenStream {
    format!("compile_error!(\"#[attr_error({})] {}\");", args, item).parse().unwrap()
}

#[proc_macro_derive(DeriveEmpty)]
pub fn derive_empty(_item: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_derive(DerivePanic)]
pub fn derive_panic(item: TokenStream) -> TokenStream {
    panic!("#[derive(DerivePanic)] {}", item);
}

#[proc_macro_derive(DeriveError)]
pub fn derive_error(item: TokenStream) -> TokenStream {
    format!("compile_error!(\"#[derive(DeriveError)] {}\");", item).parse().unwrap()
}
