extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn a_proc_macro(_item: TokenStream) -> TokenStream {
    "fn ex() { foobar::f(); }".parse().unwrap()
}

// inserts foobar::f() to the end of the function
#[proc_macro_attribute]
pub fn an_attr_macro(attr: TokenStream, item: TokenStream) -> TokenStream {
    let new_call: TokenStream = "foobar::f();".parse().unwrap();

    let mut tokens = item.into_iter();

    let fn_tok = tokens.next().unwrap();
    let ident_tok = tokens.next().unwrap();
    let args_tok = tokens.next().unwrap();
    let body = match tokens.next().unwrap() {
        TokenTree::Group(g) => {
            let new_g = Group::new(g.delimiter(), new_call);
            let mut outer_g = Group::new(
                g.delimiter(),
                [TokenTree::Group(g.clone()), TokenTree::Group(new_g)].into_iter().collect(),
            );

            if attr.to_string() == "with_span" {
                outer_g.set_span(g.span());
            }

            TokenTree::Group(outer_g)
        }
        _ => unreachable!(),
    };

    let tokens = vec![fn_tok, ident_tok, args_tok, body].into_iter().collect::<TokenStream>();

    tokens
}
