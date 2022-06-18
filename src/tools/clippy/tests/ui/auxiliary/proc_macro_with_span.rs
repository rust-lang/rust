// compile-flags: --emit=link
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{token_stream::IntoIter, Group, Span, TokenStream, TokenTree};

#[proc_macro]
pub fn with_span(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let span = iter.next().unwrap().span();
    let mut res = TokenStream::new();
    write_with_span(span, iter, &mut res);
    res
}

fn write_with_span(s: Span, input: IntoIter, out: &mut TokenStream) {
    for mut tt in input {
        if let TokenTree::Group(g) = tt {
            let mut stream = TokenStream::new();
            write_with_span(s, g.stream().into_iter(), &mut stream);
            let mut group = Group::new(g.delimiter(), stream);
            group.set_span(s);
            out.extend([TokenTree::Group(group)]);
        } else {
            tt.set_span(s);
            out.extend([tt]);
        }
    }
}
