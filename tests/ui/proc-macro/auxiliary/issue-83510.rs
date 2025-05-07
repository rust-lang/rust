extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn dance_like_you_want_to_ice(_: TokenStream) -> TokenStream {
    r#"
    impl Foo {
        type Bar = Box<()> + Baz;
    }
    "#
    .parse()
    .unwrap()
}
