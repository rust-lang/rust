/* macro namespace. */

extern crate proc_macro;
use proc_macro::*;
use std::str::FromStr;

const ERROR: &str = "fn helper() { \"helper\" }";
// https://doc.rust-lang.org/nightly/std/prelude/v1/index.html#attributes
// NOTE: all the bang macros in std are currently unstable.
#[proc_macro_attribute] pub fn test       // lang.
    (_: TokenStream, _: TokenStream) -> TokenStream {
        TokenStream::from_str("fn test_macro() { \"\" }").unwrap() }
// https://doc.rust-lang.org/nightly/reference/attributes.html#built-in-attributes-index
#[proc_macro_attribute] pub fn global_allocator // lang.
    (_: TokenStream, _: TokenStream) -> TokenStream {
        TokenStream::from_str("fn global_allocator_macro() { \"\" }").unwrap() }
