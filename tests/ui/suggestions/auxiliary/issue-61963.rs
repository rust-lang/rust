extern crate proc_macro;

use proc_macro::{Group, Spacing, Punct, TokenTree, TokenStream};

// This macro exists as part of a reproduction of #61963 but without using quote/syn/proc_macro2.

#[proc_macro_attribute]
pub fn dom_struct(_: TokenStream, input: TokenStream) -> TokenStream {
    // Construct the expected output tokens - the input but with a `#[derive(DomObject)]` applied.
    let attributes: TokenStream =
        "#[derive(DomObject)]".to_string().parse().unwrap();
    let output: TokenStream = attributes.into_iter()
        .chain(input.into_iter()).collect();

    let mut tokens: Vec<_> = output.into_iter().collect();
    // Adjust the spacing of `>` tokens to match what `quote` would produce.
    for token in tokens.iter_mut() {
        if let TokenTree::Group(group) = token {
            let mut tokens: Vec<_> = group.stream().into_iter().collect();
            for token in tokens.iter_mut() {
                if let TokenTree::Punct(p) = token {
                    if p.as_char() == '>' {
                        *p = Punct::new('>', Spacing::Alone);
                    }
                }
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
