// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The compiler code necessary to support the cfg! extension, which expands to
/// a literal `true` or `false` based on whether the given cfg matches the
/// current compilation environment.

use ast;
use codemap::Span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use attr;
use attr::*;
use parse::attr::ParserAttr;
use parse::token::{self,InternedString};
use ptr::P;

pub fn expand_cfg<'cx>(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'static> {

    let cfg = match get_cfg_meta_item(cx, sp, tts) {
        Some(cfg) => cfg,
        None => { return DummyResult::expr(sp); },
    };

    let matches_cfg = attr::cfg_matches(&cx.parse_sess.span_diagnostic, &cx.cfg, &*cfg);
    MacEager::expr(cx.expr_bool(sp, matches_cfg))
}

pub fn expand_cfg_int(cx: &mut ExtCtxt,
                      sp: Span,
                      tts: &[ast::TokenTree])
                      -> Box<base::MacResult+'static> {

    match expand_cfg_val(cx, sp, tts) {
        Some(val) => {
            match val.parse::<i64>() {
                Ok(i) => {

                    let sign = if i < 0 { ast::Minus } else { ast::Plus };
                    let magnitude = i.abs() as u64;

                    let ty = ast::UnsuffixedIntLit(sign);
                    let lit = ast::LitInt(magnitude, ty);

                    MacEager::expr(cx.expr_lit(sp, lit))
                },
                Err(..) => {
                    cx.span_err(sp, &format!("{} is not an integer", val));
                    DummyResult::expr(sp)
                },
            }
        },
        None => {
            DummyResult::expr(sp)
        }
    }
}

pub fn expand_cfg_str(cx: &mut ExtCtxt,
                      sp: Span,
                      tts: &[ast::TokenTree])
                      -> Box<base::MacResult+'static> {

    match expand_cfg_val(cx, sp, tts) {
        Some(val) => MacEager::expr(cx.expr_str(sp, val)),
        None => DummyResult::expr(sp),
    }
}

pub fn expand_cfg_val(cx: &mut ExtCtxt,
                      sp: Span,
                      tts: &[ast::TokenTree])
                      -> Option<InternedString> {

    let cfg = match get_cfg_meta_item(cx, sp, tts) {
        Some(cfg) => cfg,
        None => { return None; },
    };

    match cfg.node {
        ast::MetaList(..) |
        ast::MetaNameValue(..) => {
            cx.span_err(cfg.span, "expected a single configuration flag name");
            return None;
        },
        ast::MetaWord(ref cfg_name) => {

            // Look for the CFG meta item with the specified name.
            let item = cx.cfg.iter().find(|item| {
                if let ast::MetaNameValue(ref name,_) = item.node {
                    cfg_name == name
                } else {
                    false
                }
            });

            match item.map(|ref a| &a.node) {
                // The CFG was found
                Some(&ast::MetaNameValue(_, ref lit))=> {
                    if let ast::LitStr(ref val, _) = lit.node {
                        Some(val.clone())
                    } else {
                        // CFG values are always textual
                        unreachable!();
                    }
                },
                Some(_) => {
                    unreachable!();
                },
                None => { // the cfg is not defined
                    cx.span_err(cfg.span, "configuration flag is not set");
                    None
                }
            }
        }
    }
}

fn get_cfg_meta_item(cx: &mut ExtCtxt,
                     sp: Span,
                     tts: &[ast::TokenTree])
                     -> Option<P<ast::MetaItem>> {
    let mut p = cx.new_parser_from_tts(tts);
    let cfg = p.parse_meta_item();

    if !panictry!(p.eat(&token::Eof)){
        cx.span_err(sp, "expected 1 cfg-pattern");
        None
    } else {
        Some(cfg)
    }

} 
