// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn expect_let(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "let string = \"Hello, world!\" ;");
    item
}

#[proc_macro_attribute]
pub fn expect_my_macro_stmt(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "my_macro ! (\"{}\", string) ;");
    item
}

#[proc_macro_attribute]
pub fn expect_expr(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "print_str(\"string\")");
    item
}

#[proc_macro_attribute]
pub fn expect_my_macro_expr(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "my_macro ! (\"{}\", string)");
    item
}

#[proc_macro_attribute]
pub fn duplicate(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    format!("{}, {}", item, item).parse().unwrap()
}

#[proc_macro_attribute]
pub fn no_output(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert!(!item.to_string().is_empty());
    "".parse().unwrap()
}
