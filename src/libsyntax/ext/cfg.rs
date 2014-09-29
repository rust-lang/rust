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
use codemap::Span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use attr;
use attr::*;
use parse::attr::ParserAttr;
use parse::token;


pub fn expand_cfg<'cx>(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'static> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut cfgs = Vec::new();
    // parse `cfg!(meta_item, meta_item(x,y), meta_item="foo", ...)`
    while p.token != token::EOF {
        cfgs.push(p.parse_meta_item());
        if p.eat(&token::EOF) { break } // trailing comma is optional,.
        p.expect(&token::COMMA);
    }

    if cfgs.len() != 1 {
        cx.span_warn(sp, "The use of multiple cfgs at the top level of `cfg!` \
                          is deprecated. Change `cfg!(a, b)` to \
                          `cfg!(all(a, b))`.");
    }

    let matches_cfg = cfgs.iter().all(|cfg| attr::cfg_matches(&cx.parse_sess.span_diagnostic,
                                                              cx.cfg.as_slice(), &**cfg));

    MacExpr::new(cx.expr_bool(sp, matches_cfg))
}
