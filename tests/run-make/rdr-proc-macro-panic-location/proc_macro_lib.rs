extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(PanicDerive)]
pub fn derive_panic(_input: TokenStream) -> TokenStream {
    r#"
        impl PanicStruct {
            pub fn do_panic() {
                panic!("panic from derived impl");
            }
        }
    "#
    .parse()
    .unwrap()
}

#[proc_macro_attribute]
pub fn panic_attr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_str = item.to_string();
    format!(
        r#"
        {item_str}
        fn generated_panic() {{
            panic!("panic from attribute macro");
        }}
        "#
    )
    .parse()
    .unwrap()
}
