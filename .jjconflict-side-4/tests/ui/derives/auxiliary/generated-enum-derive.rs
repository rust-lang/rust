extern crate proc_macro;

use proc_macro::TokenStream;

// Like a compile-time generator, emit a macro whose output uses the attribute's call site.
#[proc_macro_attribute]
pub fn generator(_: TokenStream, _: TokenStream) -> TokenStream {
    "macro_rules! gen_enums_from_list {
        ([\"X\"]) => { enum Position1 { X } };
    }"
    .parse()
    .unwrap()
}

#[proc_macro]
pub fn generated_enum(_: TokenStream) -> TokenStream {
    "enum Generated { X }".parse().unwrap()
}

#[proc_macro_attribute]
pub fn passthrough(_: TokenStream, item: TokenStream) -> TokenStream {
    item
}
