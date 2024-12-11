#![feature(proc_macro_span, proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{quote, Span, TokenStream, TokenTree};

// This macro generates a macro with the same macro definition as `manual_foo` in
// `same-sequence-span.rs` but with the same span for all sequences.
#[proc_macro]
pub fn make_foo(_: TokenStream) -> TokenStream {
    let result = quote! {
        macro_rules! generated_foo {
            (1 $$x:expr $$($$y:tt,)* $$(= $$z:tt)*) => {};
        }
    };

    result
}
