//@ edition: 2024

extern crate proc_macro;

use proc_macro::{Delimiter, TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn norepr(_: TokenStream, input: TokenStream) -> TokenStream {
    let mut tokens = vec![];
    let mut tts = input.into_iter().fuse().peekable();
    loop {
        let Some(token) = tts.next() else { break };
        if let TokenTree::Punct(punct) = &token
            && punct.as_char() == '#'
        {
            if let Some(TokenTree::Group(group)) = tts.peek()
                && let Delimiter::Bracket = group.delimiter()
                && let Some(TokenTree::Ident(ident)) = group.stream().into_iter().next()
                && ident.to_string() == "repr"
            {
                let _ = tts.next();
                // skip '#' and '[repr(..)]
            } else {
                tokens.push(token);
            }
        } else {
            tokens.push(token);
        }
    }
    tokens.into_iter().collect()
}
