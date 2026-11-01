//@ edition: 2024
// (The proc-macro crate doesn't need to be instrumented.)
//@ compile-flags: -Cinstrument-coverage=off

use proc_macro::TokenStream;

/// Minimized form of `#[derive(arbitrary::Arbitrary)]` that still triggers
/// the original bug.
const CODE: &str = "
    impl Arbitrary for MyEnum {
        fn try_size_hint() -> Option<usize> {
            Some(0)?;
            None
        }
    }
";

#[proc_macro_attribute]
pub fn attr(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    CODE.parse().unwrap()
}

#[proc_macro]
pub fn bang(_item: TokenStream) -> TokenStream {
    CODE.parse().unwrap()
}

#[proc_macro_derive(Arbitrary)]
pub fn derive_arbitrary(_item: TokenStream) -> TokenStream {
    CODE.parse().unwrap()
}
