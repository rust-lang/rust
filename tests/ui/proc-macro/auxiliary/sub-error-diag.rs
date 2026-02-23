#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::{Diagnostic, Level, Span};

#[proc_macro_attribute]
pub fn proc_emit_err(
    _: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    Diagnostic::new(Level::Error, "Parent message")
        .span_error(Span::call_site(), "Child message")
        .emit();

    input
}
