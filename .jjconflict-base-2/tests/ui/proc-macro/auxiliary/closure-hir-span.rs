//@ edition:2024

extern crate proc_macro;

#[proc_macro]
pub fn m(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut iter = input.into_iter();
    let move_token = iter.next().unwrap();
    let pipe1 = iter.next().unwrap();
    let pipe2 = iter.next().unwrap();
    let body = iter.next().unwrap();

    let mut inner = proc_macro::TokenStream::new();
    inner.extend([body]);
    let new_body = proc_macro::TokenTree::Group(proc_macro::Group::new(
        proc_macro::Delimiter::Brace,
        inner,
    ));

    let mut out = proc_macro::TokenStream::new();
    out.extend([move_token, pipe1, pipe2, new_body]);
    out
}
