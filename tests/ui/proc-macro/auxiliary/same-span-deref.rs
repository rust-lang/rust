extern crate proc_macro;
use proc_macro::{Punct, Spacing, TokenStream, TokenTree};

#[proc_macro]
pub fn same_span_deref(input: TokenStream) -> TokenStream {
    let mut result = Vec::new();

    let first_token = input.clone().into_iter().next().expect("need at least one token");
    let span = first_token.span();

    // Create a `*` with the same span as the input expression.
    // This makes expr.span.lo == inner.span.lo, so `expr.span.until(inner.span)`
    // produces an empty span.
    let mut star = Punct::new('*', Spacing::Alone);
    star.set_span(span);

    result.push(TokenTree::Punct(star));
    result.extend(input);

    result.into_iter().collect()
}
