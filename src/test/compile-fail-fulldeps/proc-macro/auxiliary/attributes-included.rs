// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// no-prefer-dynamic

#![feature(proc_macro)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, TokenNode, Delimiter, Literal, Spacing};

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
    match slice[0].kind {
        TokenNode::Op('#', _) => {}
        _ => panic!("expected '#' char"),
    }
    match slice[1].kind {
        TokenNode::Group(Delimiter::Bracket, _) => {}
        _ => panic!("expected brackets"),
    }
    *slice = &slice[2..];
}

fn assert_doc(slice: &mut &[TokenTree]) {
    match slice[0].kind {
        TokenNode::Op('#', Spacing::Alone) => {}
        _ => panic!("expected #"),
    }
    let inner = match slice[1].kind {
        TokenNode::Group(Delimiter::Bracket, ref s) => s.clone(),
        _ => panic!("expected brackets"),
    };
    let tokens = inner.into_iter().collect::<Vec<_>>();
    let tokens = &tokens[..];

    if tokens.len() != 3 {
        panic!("expected three tokens in doc")
    }

    match tokens[0].kind {
        TokenNode::Term(ref t) => assert_eq!("doc", t.as_str()),
        _ => panic!("expected `doc`"),
    }
    match tokens[1].kind {
        TokenNode::Op('=', Spacing::Alone) => {}
        _ => panic!("expected equals"),
    }
    match tokens[2].kind {
        TokenNode::Literal(_) => {}
        _ => panic!("expected literal"),
    }

    *slice = &slice[2..];
}

fn assert_invoc(slice: &mut &[TokenTree]) {
    match slice[0].kind {
        TokenNode::Op('#', _) => {}
        _ => panic!("expected '#' char"),
    }
    match slice[1].kind {
        TokenNode::Group(Delimiter::Bracket, _) => {}
        _ => panic!("expected brackets"),
    }
    *slice = &slice[2..];
}

fn assert_foo(slice: &mut &[TokenTree]) {
    match slice[0].kind {
        TokenNode::Term(ref name) => assert_eq!(name.as_str(), "fn"),
        _ => panic!("expected fn"),
    }
    match slice[1].kind {
        TokenNode::Term(ref name) => assert_eq!(name.as_str(), "foo"),
        _ => panic!("expected foo"),
    }
    match slice[2].kind {
        TokenNode::Group(Delimiter::Parenthesis, ref s) => assert!(s.is_empty()),
        _ => panic!("expected parens"),
    }
    match slice[3].kind {
        TokenNode::Group(Delimiter::Brace, _) => {}
        _ => panic!("expected braces"),
    }
    *slice = &slice[4..];
}

fn fold_stream(input: TokenStream) -> TokenStream {
    input.into_iter().map(fold_tree).collect()
}

fn fold_tree(input: TokenTree) -> TokenTree {
    TokenTree {
        span: input.span,
        kind: fold_node(input.kind),
    }
}

fn fold_node(input: TokenNode) -> TokenNode {
    match input {
        TokenNode::Group(a, b) => TokenNode::Group(a, fold_stream(b)),
        TokenNode::Op(a, b) => TokenNode::Op(a, b),
        TokenNode::Term(a) => TokenNode::Term(a),
        TokenNode::Literal(a) => {
            if a.to_string() != "\"foo\"" {
                TokenNode::Literal(a)
            } else {
                TokenNode::Literal(Literal::integer(3))
            }
        }
    }
}
