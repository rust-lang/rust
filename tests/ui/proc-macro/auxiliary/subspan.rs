#![feature(proc_macro_diagnostic, proc_macro_span)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Span, Diagnostic};

fn parse(input: TokenStream) -> Result<(), Diagnostic> {
    if let Some(TokenTree::Literal(lit)) = input.into_iter().next() {
        let mut spans = vec![];
        let string = lit.to_string();
        for hi in string.matches("hi") {
            let index = hi.as_ptr() as usize - string.as_ptr() as usize;
            let subspan = lit.subspan(index..(index + hi.len())).unwrap();
            spans.push(subspan);
        }

        if !spans.is_empty() {
            Err(Span::call_site().error("found 'hi's").span_note(spans, "here"))
        } else {
            Ok(())
        }
    } else {
        Err(Span::call_site().error("invalid input: expected string literal"))
    }
}

#[proc_macro]
pub fn subspan(input: TokenStream) -> TokenStream {
    if let Err(diag) = parse(input) {
        diag.emit();
    }

    TokenStream::new()
}
