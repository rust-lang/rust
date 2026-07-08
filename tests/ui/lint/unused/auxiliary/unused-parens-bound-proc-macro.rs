extern crate proc_macro;

use proc_macro::{Group, Span, TokenStream, TokenTree};

// Recursively overwrite the span of every token (including group delimiters)
// with `span`.
fn respan(span: Span, stream: TokenStream) -> TokenStream {
    stream
        .into_iter()
        .map(|tt| match tt {
            TokenTree::Group(group) => {
                let mut group = Group::new(group.delimiter(), respan(span, group.stream()));
                group.set_span(span);
                TokenTree::Group(group)
            }
            mut tt => {
                tt.set_span(span);
                tt
            }
        })
        .collect()
}

/// Emits `const _: &dyn (Send) = &();` with every token carrying the span of the
/// macro's first input token. The parenthesized trait-object bound is therefore
/// reported at a span that does not actually contain parentheses in the source.
#[proc_macro]
pub fn emit_parenthesized_bound(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();
    let code: TokenStream = "const _: &dyn (Send) = &();".parse().unwrap();
    respan(span, code)
}
