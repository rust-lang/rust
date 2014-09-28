// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use attr;
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ptr::P;

pub fn expand(cx: &mut ExtCtxt, sp: Span, mi: &ast::MetaItem, it: P<ast::Item>) -> P<ast::Item> {
    let (cfg, attr) = match mi.node {
        ast::MetaList(_, ref mis) if mis.len() == 2 => (&mis[0], &mis[1]),
        _ => {
            cx.span_err(sp, "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
            return it;
        }
    };

    let mut out = (*it).clone();
    if attr::cfg_matches(&cx.parse_sess.span_diagnostic, cx.cfg.as_slice(), &**cfg) {
        out.attrs.push(cx.attribute(attr.span, attr.clone()));
    }

    P(out)
}

