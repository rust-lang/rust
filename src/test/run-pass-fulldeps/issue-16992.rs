// ignore-cross-compile

#![feature(quote, rustc_private)]

extern crate syntax;

use syntax::ext::base::ExtCtxt;

#[allow(dead_code)]
fn foobar(cx: &mut ExtCtxt) {
    quote_expr!(cx, 1);
    quote_expr!(cx, 2);
}

fn main() { }
