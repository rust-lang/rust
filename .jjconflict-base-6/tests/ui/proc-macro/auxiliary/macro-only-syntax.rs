// These are tests for syntax that is accepted by the Rust parser but
// unconditionally rejected semantically after macro expansion. Attribute macros
// are permitted to accept such syntax as long as they replace it with something
// that makes sense to Rust.
//
// We also inspect some of the spans to verify the syntax is not triggering the
// lossy string reparse hack (https://github.com/rust-lang/rust/issues/43081).

#![feature(proc_macro_span)]

extern crate proc_macro;
use proc_macro::{token_stream, Delimiter, TokenStream, TokenTree};
use std::path::Component;

// unsafe mod m {
//     pub unsafe mod inner;
// }
#[proc_macro_attribute]
pub fn expect_unsafe_mod(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let tokens = &mut input.into_iter();
    expect(tokens, "unsafe");
    expect(tokens, "mod");
    expect(tokens, "m");
    let tokens = &mut expect_brace(tokens);
    expect(tokens, "pub");
    expect(tokens, "unsafe");
    expect(tokens, "mod");
    let ident = expect(tokens, "inner");
    expect(tokens, ";");
    check_useful_span(ident, "unsafe-mod.rs");
    TokenStream::new()
}

// unsafe extern {
//     type T;
// }
#[proc_macro_attribute]
pub fn expect_unsafe_foreign_mod(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let tokens = &mut input.into_iter();
    expect(tokens, "unsafe");
    expect(tokens, "extern");
    let tokens = &mut expect_brace(tokens);
    expect(tokens, "type");
    let ident = expect(tokens, "T");
    expect(tokens, ";");
    check_useful_span(ident, "unsafe-foreign-mod.rs");
    TokenStream::new()
}

// unsafe extern "C++" {}
#[proc_macro_attribute]
pub fn expect_unsafe_extern_cpp_mod(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let tokens = &mut input.into_iter();
    expect(tokens, "unsafe");
    expect(tokens, "extern");
    let abi = expect(tokens, "\"C++\"");
    expect_brace(tokens);
    check_useful_span(abi, "unsafe-foreign-mod.rs");
    TokenStream::new()
}

fn expect(tokens: &mut token_stream::IntoIter, expected: &str) -> TokenTree {
    match tokens.next() {
        Some(token) if token.to_string() == expected => token,
        wrong => panic!("unexpected token: {:?}, expected `{}`", wrong, expected),
    }
}

fn expect_brace(tokens: &mut token_stream::IntoIter) -> token_stream::IntoIter {
    match tokens.next() {
        Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Brace => {
            group.stream().into_iter()
        }
        wrong => panic!("unexpected token: {:?}, expected `{{`", wrong),
    }
}

fn check_useful_span(token: TokenTree, expected_filename: &str) {
    let span = token.span();
    assert!(span.column() < span.end().column());

    let source_path = span.local_file().unwrap();
    let filename = source_path.components().last().unwrap();
    assert_eq!(filename, Component::Normal(expected_filename.as_ref()));
}
