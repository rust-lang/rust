extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn assert1(_a: TokenStream, b: TokenStream) -> TokenStream {
    assert_eq(b.clone(), "pub fn foo() {}".parse().unwrap());
    b
}

#[proc_macro_derive(Foo, attributes(foo))]
pub fn assert2(a: TokenStream) -> TokenStream {
    assert_eq(a, "pub struct MyStructc { _a: i32, }".parse().unwrap());
    TokenStream::new()
}

fn assert_eq(a: TokenStream, b: TokenStream) {
    let mut a = a.into_iter();
    let mut b = b.into_iter();
    for (a, b) in a.by_ref().zip(&mut b) {
        match (a, b) {
            (TokenTree::Group(a), TokenTree::Group(b)) => {
                assert_eq!(a.delimiter(), b.delimiter());
                assert_eq(a.stream(), b.stream());
            }
            (TokenTree::Punct(a), TokenTree::Punct(b)) => {
                assert_eq!(a.as_char(), b.as_char());
                assert_eq!(a.spacing(), b.spacing());
            }
            (TokenTree::Literal(a), TokenTree::Literal(b)) => {
                assert_eq!(a.to_string(), b.to_string());
            }
            (TokenTree::Ident(a), TokenTree::Ident(b)) => {
                assert_eq!(a.to_string(), b.to_string());
            }
            (a, b) => panic!("{:?} != {:?}", a, b),
        }
    }

    assert!(a.next().is_none());
    assert!(b.next().is_none());
}
