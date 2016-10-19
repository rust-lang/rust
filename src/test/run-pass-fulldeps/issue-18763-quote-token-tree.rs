// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
