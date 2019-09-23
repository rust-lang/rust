// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_span, proc_macro_hygiene, proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{quote, Span, TokenStream, TokenTree};

fn assert_same_span(a: Span, b: Span) {
    assert_eq!(a.start(), b.start());
    assert_eq!(a.end(), b.end());
}

// This macro generates a macro with the same macro definition as `manual_foo` in
// `same-sequence-span.rs` but with the same span for all sequences.
#[proc_macro]
pub fn make_foo(_: TokenStream) -> TokenStream {
    let result = quote! {
        macro_rules! generated_foo {
            (1 $$x:expr $$($$y:tt,)* $$(= $$z:tt)*) => {};
        }
    };

    // Check that all spans are equal.
    // FIXME: `quote!` gives def-site spans to idents and literals,
    // but leaves (default) call-site spans on groups and punctuation.
    let mut span_call = None;
    let mut span_def = None;
    for tt in result.clone() {
        match tt {
            TokenTree::Ident(..) | TokenTree::Literal(..) => match span_def {
                None => span_def = Some(tt.span()),
                Some(span) => assert_same_span(tt.span(), span),
            }
            TokenTree::Punct(..) | TokenTree::Group(..) => match span_call {
                None => span_call = Some(tt.span()),
                Some(span) => assert_same_span(tt.span(), span),
            }
        }

    }

    result
}
