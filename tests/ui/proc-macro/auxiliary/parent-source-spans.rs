// force-host
// no-prefer-dynamic

#![feature(proc_macro_diagnostic, proc_macro_span)]
#![crate_type = "proc-macro"]

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
pub fn parent_source_spans(input: TokenStream) -> TokenStream {
    let mut tokens = input.into_iter();
    let (sp1, str1) = lit_span(tokens.next().expect("first string"));
    let _ = tokens.next();
    let (sp2, str2) = lit_span(tokens.next().expect("second string"));

    sp1.error(format!("first final: {}", str1)).emit();
    sp2.error(format!("second final: {}", str2)).emit();

    if let (Some(p1), Some(p2)) = (sp1.parent(), sp2.parent()) {
        p1.error(format!("first parent: {}", str1)).emit();
        p2.error(format!("second parent: {}", str2)).emit();

        if let (Some(gp1), Some(gp2)) = (p1.parent(), p2.parent()) {
            gp1.error(format!("first grandparent: {}", str1)).emit();
            gp2.error(format!("second grandparent: {}", str2)).emit();
        }
    }

    sp1.source().error(format!("first source: {}", str1)).emit();
    sp2.source().error(format!("second source: {}", str2)).emit();

    "ok".parse().unwrap()
}
