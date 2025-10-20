extern crate proc_macro;
use proc_macro::*;

fn spans_callsite(ts: TokenStream) -> TokenStream {
    let mut new_ts = TokenStream::new();

    for i in ts {
        let new_token = i.clone();
        match new_token {
            TokenTree::Group(g) => {
                new_ts.extend([Group::new(g.delimiter(), spans_callsite(g.stream()))])
            }
            mut other => {
                other.set_span(Span::call_site());
                new_ts.extend([other]);
            }
        }
    }

    new_ts
}

#[proc_macro_attribute]
pub fn all_spans_same(_: TokenStream, ts: TokenStream) -> TokenStream {
    spans_callsite(ts)
}
