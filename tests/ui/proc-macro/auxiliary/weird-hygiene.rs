extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Group};

fn find_my_ident(tokens: TokenStream) -> Option<TokenStream> {
    for token in tokens {
        if let TokenTree::Ident(ident) = &token {
            if ident.to_string() == "hidden_ident" {
                return Some(vec![token].into_iter().collect())
            }
        } else if let TokenTree::Group(g) = token {
            if let Some(stream) = find_my_ident(g.stream()) {
                return Some(stream)
            }
        }
    }
    return None;
}


#[proc_macro_derive(WeirdDerive)]
pub fn weird_derive(item: TokenStream) -> TokenStream {
    let my_ident = find_my_ident(item).expect("Missing 'my_ident'!");
    let tokens: TokenStream = "call_it!();".parse().unwrap();
    let final_call = tokens.into_iter().map(|tree| {
        if let TokenTree::Group(g) = tree {
            return Group::new(g.delimiter(), my_ident.clone()).into()
        } else {
            return tree
        }
    }).collect();
    final_call
}

#[proc_macro]
pub fn recollect(item: TokenStream) -> TokenStream {
    item.into_iter().collect()
}

#[proc_macro_attribute]
pub fn recollect_attr(_attr: TokenStream, mut item: TokenStream) -> TokenStream {
    item.into_iter().collect()
}
