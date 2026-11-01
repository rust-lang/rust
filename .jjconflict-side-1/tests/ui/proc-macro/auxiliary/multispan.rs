#![feature(proc_macro_diagnostic, proc_macro_span, proc_macro_def_site)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Span, Diagnostic};

fn parse(input: TokenStream) -> Result<(), Diagnostic> {
    let mut hi_spans = vec![];
    for tree in input {
        if let TokenTree::Ident(ref ident) = tree {
            if ident.to_string() == "hi" {
                hi_spans.push(ident.span());
            }
        }
    }

    if !hi_spans.is_empty() {
        return Err(Span::def_site()
                       .error("hello to you, too!")
                       .span_note(hi_spans, "found these 'hi's"));
    }

    Ok(())
}

#[proc_macro]
pub fn hello(input: TokenStream) -> TokenStream {
    if let Err(diag) = parse(input) {
        diag.emit();
    }

    TokenStream::new()
}
