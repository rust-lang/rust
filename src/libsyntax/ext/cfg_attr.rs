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
    if cfg_matches(cx, &**cfg) {
        out.attrs.push(cx.attribute(attr.span, attr.clone()));
    }

    P(out)
}

fn cfg_matches(cx: &mut ExtCtxt, cfg: &ast::MetaItem) -> bool {
    match cfg.node {
        ast::MetaList(ref pred, ref mis) if pred.get() == "any" =>
            mis.iter().any(|mi| cfg_matches(cx, &**mi)),
        ast::MetaList(ref pred, ref mis) if pred.get() == "all" =>
            mis.iter().all(|mi| cfg_matches(cx, &**mi)),
        ast::MetaList(ref pred, ref mis) if pred.get() == "not" => {
            if mis.len() != 1 {
                cx.span_err(cfg.span, format!("expected 1 value, got {}",
                                              mis.len()).as_slice());
                return false;
            }
            !cfg_matches(cx, &*mis[0])
        }
        ast::MetaList(ref pred, _) => {
            cx.span_err(cfg.span,
                        format!("invalid predicate `{}`", pred).as_slice());
            false
        },
        ast::MetaWord(_) | ast::MetaNameValue(..) =>
            attr::contains(cx.cfg.as_slice(), cfg),
    }
}
