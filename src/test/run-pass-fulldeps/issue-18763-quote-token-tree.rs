#![allow(dead_code)]
#![allow(unused_imports)]
// ignore-cross-compile
#![feature(quote, rustc_private)]

extern crate syntax;

use syntax::ext::base::ExtCtxt;

fn syntax_extension(cx: &ExtCtxt) {
    let _toks_1 = vec![quote_tokens!(cx, /** comment */ fn foo() {})];
    let name = quote_tokens!(cx, bar);
    let _toks_2 = vec![quote_item!(cx, static $name:isize = 2;)];
    let _toks_4 = quote_tokens!(cx, $name:static $name:sizeof);
    let _toks_3 = vec![quote_item!(cx,
        /// comment
        fn foo() { let $name:isize = 3; }
    )];
}

fn main() {
}
