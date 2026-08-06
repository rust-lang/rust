extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, TokenStream, TokenTree};

#[proc_macro_derive(Test)]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = input.into_iter().collect::<Vec<_>>();
    let input_span = input[0].span();

    let output = "impl MyEnum {
        pub(crate) fn false_positive(self) -> i8 {
            self as i8
        }
    }"
    .parse::<TokenStream>()
    .unwrap()
    .into_iter()
    .collect::<Vec<_>>();

    let TokenTree::Group(body) = &output[2] else { unreachable!() };
    let mut body = body.stream().into_iter().collect::<Vec<_>>();

    // Preserve the visibility tokens from the input, as `quote!` does when interpolating `#vis`.
    body[0] = input[0].clone();
    body[1] = input[1].clone();

    // Give the return type the input's span, matching `Ident::new("i8", input.span())`.
    body[7] = TokenTree::Ident(Ident::new("i8", input_span));

    let mut generated_body =
        Group::new(Delimiter::Brace, body.into_iter().collect::<TokenStream>());
    generated_body.set_span(output[2].span());

    vec![output[0].clone(), output[1].clone(), generated_body.into()].into_iter().collect()
}
