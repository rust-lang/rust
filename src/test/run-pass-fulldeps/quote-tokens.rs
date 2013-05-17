// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_implicitly_copyable_typarams)];

extern mod syntax;

use syntax::ext::base::ExtCtxt;

fn syntax_extension(ext_cx: @ExtCtxt) {
    let e_toks : ~[syntax::ast::token_tree] = quote_tokens!(1 + 2);
    let p_toks : ~[syntax::ast::token_tree] = quote_tokens!((x, 1 .. 4, *));

    let a: @syntax::ast::expr = quote_expr!(1 + 2);
    let _b: Option<@syntax::ast::item> = quote_item!( static foo : int = $e_toks; );
    let _c: @syntax::ast::pat = quote_pat!( (x, 1 .. 4, *) );
    let _d: @syntax::ast::stmt = quote_stmt!( let x = $a; );
    let _e: @syntax::ast::expr = quote_expr!( match foo { $p_toks => 10 } );
}

fn main() {
}
