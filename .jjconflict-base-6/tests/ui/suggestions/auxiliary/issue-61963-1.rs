extern crate proc_macro;

use proc_macro::{Group, TokenStream, TokenTree};

// This macro exists as part of a reproduction of #61963 but without using quote/syn/proc_macro2.

#[proc_macro_derive(DomObject)]
pub fn expand_token_stream(input: TokenStream) -> TokenStream {
    // Construct a dummy span - `#0 bytes(0..0)` - which is present in the input because
    // of the specially crafted generated tokens in the `attribute-crate` proc-macro.
    let dummy_span = input.clone().into_iter().nth(0).unwrap().span();

    // Define what the macro would output if constructed properly from the source using syn/quote.
    let output: TokenStream = "impl Bar for ((), Qux<Qux<Baz> >) { }
    impl Bar for ((), Box<Bar>) { }".parse().unwrap();

    let mut tokens: Vec<_> = output.into_iter().collect();
    // Adjust token spans to match the original crate (which would use `quote`). Some of the
    // generated tokens point to the dummy span.
    for token in tokens.iter_mut() {
        if let TokenTree::Group(group) = token {
            let mut tokens: Vec<_> = group.stream().into_iter().collect();
            for token in tokens.iter_mut().skip(2) {
                token.set_span(dummy_span);
            }

            let mut stream = TokenStream::new();
            stream.extend(tokens);
            *group = Group::new(group.delimiter(), stream);
        }
    }

    let mut output = TokenStream::new();
    output.extend(tokens);
    output
}
