extern crate proc_macro;

use proc_macro::TokenStream;

fn compile_error() -> TokenStream {
    r#"compile_error!("")"#.parse().unwrap()
}

#[proc_macro_derive(MyTrait)]
pub fn derive(input: TokenStream) -> TokenStream {
    compile_error()
}
#[proc_macro_attribute]
pub fn attribute_macro(_attr: TokenStream, mut input: TokenStream) -> TokenStream {
    input.extend(compile_error());
    input
}
#[proc_macro]
pub fn fn_macro(_item: TokenStream) -> TokenStream {
    compile_error()
}
