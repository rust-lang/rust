extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn to_unit(_: TokenStream, struct_item: TokenStream) -> TokenStream {
    rewrite(struct_item, ";")
}
#[proc_macro_attribute]
pub fn to_tuple(_: TokenStream, struct_item: TokenStream) -> TokenStream {
    rewrite(struct_item, "();")
}
#[proc_macro_attribute]
pub fn to_braced(_: TokenStream, struct_item: TokenStream) -> TokenStream {
    rewrite(struct_item, "{}")
}

fn rewrite(struct_item: TokenStream, suffix: &str) -> TokenStream {
    let ident = struct_item.into_iter().nth(1).unwrap();
    format!("struct {ident}{suffix}").parse().unwrap()
}
