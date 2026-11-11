extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn matcher(input: TokenStream) -> TokenStream {
"
struct S(());
let s = S(());
match s {
    true => {}
    _ => {}
}
".parse().unwrap()
}
