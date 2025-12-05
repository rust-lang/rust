extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn attr_with_args(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = args.to_string();

    assert_eq!(args, r#"text = "Hello, world!""#);

    let input = input.to_string();

    assert_eq!(input, "fn foo() {}");

    r#"
        fn foo() -> &'static str { "Hello, world!" }
    "#.parse().unwrap()
}

#[proc_macro_attribute]
pub fn identity(attr_args: TokenStream, _: TokenStream) -> TokenStream {
    attr_args
}
