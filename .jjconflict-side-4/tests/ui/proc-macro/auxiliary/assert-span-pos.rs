#![feature(proc_macro_diagnostic, proc_macro_span)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Span};

fn lit_span(tt: TokenTree) -> (Span, String) {
    match tt {
        TokenTree::Literal(..) |
        TokenTree::Group(..) => (tt.span(), tt.to_string().trim().into()),
        _ => panic!("expected a literal in token tree, got: {:?}", tt)
    }
}

#[proc_macro]
pub fn assert_span_pos(input: TokenStream) -> TokenStream {
    let mut tokens = input.into_iter();
    let (sp1, str1) = lit_span(tokens.next().expect("first argument"));
    let _ = tokens.next();
    let (_sp2, str2) = lit_span(tokens.next().expect("second argument"));

    let line: usize = str1.parse().unwrap();
    let col: usize = str2.parse().unwrap();

    if (line, col) != (sp1.line(), sp1.column()) {
        let msg = format!("line/column mismatch: ({}, {}) != ({}, {})", line, col,
            sp1.line(), sp1.column());
        sp1.error(msg).emit();
    }

    "".parse().unwrap()
}
