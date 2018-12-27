#![allow(dead_code)]
// ignore-cross-compile
#![feature(quote, rustc_private)]
#![deny(unused_variables)]

extern crate syntax;

use syntax::ext::base::ExtCtxt;

fn test(cx: &mut ExtCtxt) {
    let foo = 10;
    let _e = quote_expr!(cx, $foo);
}

pub fn main() { }
