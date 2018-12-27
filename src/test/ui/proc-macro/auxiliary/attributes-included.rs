// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Delimiter, Literal, Spacing, Group};

#[proc_macro_attribute]
pub fn foo(attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(attr.is_empty());
    let input = input.into_iter().collect::<Vec<_>>();
    {
        let mut cursor = &input[..];
        assert_inline(&mut cursor);
        assert_doc(&mut cursor);
        assert_inline(&mut cursor);
        assert_doc(&mut cursor);
        assert_foo(&mut cursor);
        assert!(cursor.is_empty());
    }
    fold_stream(input.into_iter().collect())
}

#[proc_macro_attribute]
pub fn bar(attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(attr.is_empty());
    let input = input.into_iter().collect::<Vec<_>>();
    {
        let mut cursor = &input[..];
        assert_inline(&mut cursor);
        assert_doc(&mut cursor);
        assert_invoc(&mut cursor);
        assert_inline(&mut cursor);
        assert_doc(&mut cursor);
        assert_foo(&mut cursor);
        assert!(cursor.is_empty());
    }
    input.into_iter().collect()
}

fn assert_inline(slice: &mut &[TokenTree]) {
    match &slice[0] {
        TokenTree::Punct(tt) => assert_eq!(tt.as_char(), '#'),
        _ => panic!("expected '#' char"),
    }
    match &slice[1] {
        TokenTree::Group(tt) => assert_eq!(tt.delimiter(), Delimiter::Bracket),
        _ => panic!("expected brackets"),
    }
    *slice = &slice[2..];
}

fn assert_doc(slice: &mut &[TokenTree]) {
    match &slice[0] {
        TokenTree::Punct(tt) => {
            assert_eq!(tt.as_char(), '#');
            assert_eq!(tt.spacing(), Spacing::Alone);
        }
        _ => panic!("expected #"),
    }
    let inner = match &slice[1] {
        TokenTree::Group(tt) => {
            assert_eq!(tt.delimiter(), Delimiter::Bracket);
            tt.stream()
        }
        _ => panic!("expected brackets"),
    };
    let tokens = inner.into_iter().collect::<Vec<_>>();
    let tokens = &tokens[..];

    if tokens.len() != 3 {
        panic!("expected three tokens in doc")
    }

    match &tokens[0] {
        TokenTree::Ident(tt) => assert_eq!("doc", &*tt.to_string()),
        _ => panic!("expected `doc`"),
    }
    match &tokens[1] {
        TokenTree::Punct(tt) => {
            assert_eq!(tt.as_char(), '=');
            assert_eq!(tt.spacing(), Spacing::Alone);
        }
        _ => panic!("expected equals"),
    }
    match tokens[2] {
        TokenTree::Literal(_) => {}
        _ => panic!("expected literal"),
    }

    *slice = &slice[2..];
}

fn assert_invoc(slice: &mut &[TokenTree]) {
    match &slice[0] {
        TokenTree::Punct(tt) => assert_eq!(tt.as_char(), '#'),
        _ => panic!("expected '#' char"),
    }
    match &slice[1] {
        TokenTree::Group(tt) => assert_eq!(tt.delimiter(), Delimiter::Bracket),
        _ => panic!("expected brackets"),
    }
    *slice = &slice[2..];
}

fn assert_foo(slice: &mut &[TokenTree]) {
    match &slice[0] {
        TokenTree::Ident(tt) => assert_eq!(&*tt.to_string(), "fn"),
        _ => panic!("expected fn"),
    }
    match &slice[1] {
        TokenTree::Ident(tt) => assert_eq!(&*tt.to_string(), "foo"),
        _ => panic!("expected foo"),
    }
    match &slice[2] {
        TokenTree::Group(tt) => {
            assert_eq!(tt.delimiter(), Delimiter::Parenthesis);
            assert!(tt.stream().is_empty());
        }
        _ => panic!("expected parens"),
    }
    match &slice[3] {
        TokenTree::Group(tt) => assert_eq!(tt.delimiter(), Delimiter::Brace),
        _ => panic!("expected braces"),
    }
    *slice = &slice[4..];
}

fn fold_stream(input: TokenStream) -> TokenStream {
    input.into_iter().map(fold_tree).collect()
}

fn fold_tree(input: TokenTree) -> TokenTree {
    match input {
        TokenTree::Group(b) => {
            TokenTree::Group(Group::new(b.delimiter(), fold_stream(b.stream())))
        }
        TokenTree::Punct(b) => TokenTree::Punct(b),
        TokenTree::Ident(a) => TokenTree::Ident(a),
        TokenTree::Literal(a) => {
            if a.to_string() != "\"foo\"" {
                TokenTree::Literal(a)
            } else {
                TokenTree::Literal(Literal::i32_unsuffixed(3))
            }
        }
    }
}
