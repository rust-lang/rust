// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
The compiler code necessary to support the cfg! extension, which
expands to a literal `true` or `false` based on whether the given cfgs
match the current compilation environment.
*/

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use attr;
use attr::*;
use parse;
use parse::token;
use parse::attr::parser_attr;

pub fn expand_cfg(cx: @ExtCtxt, sp: span, tts: &[ast::token_tree]) -> base::MacResult {
    let p = parse::new_parser_from_tts(cx.parse_sess(), cx.cfg(), tts.to_owned());

    let mut cfgs = ~[];
    // parse `cfg!(meta_item, meta_item(x,y), meta_item="foo", ...)`
    while *p.token != token::EOF {
        cfgs.push(p.parse_meta_item());
        if p.eat(&token::EOF) { break } // trailing comma is optional,.
        p.expect(&token::COMMA);
    }

    // test_cfg searches for meta items looking like `cfg(foo, ...)`
    let in_cfg = &[cx.meta_list(sp, @"cfg", cfgs)];

    let matches_cfg = attr::test_cfg(cx.cfg(), in_cfg.iter().transform(|&x| x));
    let e = cx.expr_bool(sp, matches_cfg);
    MRExpr(e)
}
