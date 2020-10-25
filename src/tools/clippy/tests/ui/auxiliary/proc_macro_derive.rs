// compile-flags: --emit=link
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(repr128, proc_macro_quote)]
#![allow(incomplete_features)]
#![allow(clippy::eq_op)]

extern crate proc_macro;

use proc_macro::{quote, TokenStream};

#[proc_macro_derive(DeriveSomething)]
pub fn derive(_: TokenStream) -> TokenStream {
    // Shound not trigger `used_underscore_binding`
    let _inside_derive = 1;
    assert_eq!(_inside_derive, _inside_derive);

    let output = quote! {
        // Should not trigger `useless_attribute`
        #[allow(dead_code)]
        extern crate rustc_middle;
    };
    output
}
