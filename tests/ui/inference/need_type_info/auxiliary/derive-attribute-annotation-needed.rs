//@ edition: 2018
#![feature(proc_macro_quote)]
use proc_macro::{TokenStream, Ident, TokenTree, quote};

#[proc_macro_derive(Serialize, attributes(serde))]
pub fn serialize(input: TokenStream) -> TokenStream {
    assert_eq!(
        input.to_string(),
        "pub struct Matrix { #[serde(serialize_with = \"f\")] matrix: (), }"
    );

    let TokenTree::Group(group) = input.into_iter().last().unwrap() else {
        panic!("unexpected group");
    };

    let TokenTree::Group(group) = group.stream().into_iter().nth(1).unwrap() else {
        panic!("expected group");
    };

    let TokenTree::Group(group) = group.stream().into_iter().nth(1).unwrap() else {
        panic!("expected group");
    };

    let TokenTree::Literal(lit) = group.stream().into_iter().nth(2).unwrap() else {
        panic!("expected literal");
    };

    let ident = Ident::new("f", lit.span());

    quote! {
        fn serialize() {
            $ident();
        }
    }
    .into()
}
