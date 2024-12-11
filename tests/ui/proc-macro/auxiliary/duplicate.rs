#![deny(unused)]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn duplicate(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut new_name = Some(attr.into_iter().nth(0).unwrap());
    let mut encountered_idents = 0;
    let input = item.to_string();
    let ret = item
        .into_iter()
        .map(move |token| match token {
            TokenTree::Ident(_) if encountered_idents == 1 => {
                encountered_idents += 1;
                new_name.take().unwrap()
            }
            TokenTree::Ident(_) => {
                encountered_idents += 1;
                token
            }
            _ => token,
        })
        .collect::<TokenStream>();
    let mut input_again = input.parse::<TokenStream>().unwrap();
    input_again.extend(ret);
    input_again
}
