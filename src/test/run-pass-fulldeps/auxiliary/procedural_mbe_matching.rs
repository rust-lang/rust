// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![crate_type="dylib"]
#![feature(plugin_registrar, quote, rustc_private)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc;
extern crate rustc_plugin;

use syntax::parse::token::{NtExpr, NtPat};
use syntax::ast::{Ident, Pat};
use syntax::tokenstream::{TokenTree};
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::ext::build::AstBuilder;
use syntax::ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use syntax::ext::tt::macro_parser::{Success, Failure, Error};
use syntax::ext::tt::macro_parser::parse_failure_msg;
use syntax::ptr::P;
use syntax_pos::Span;
use rustc_plugin::Registry;

fn expand_mbe_matches(cx: &mut ExtCtxt, _: Span, args: &[TokenTree])
        -> Box<MacResult + 'static> {

    let mbe_matcher = quote_matcher!(cx, $matched:expr, $($pat:pat)|+);
    let map = match TokenTree::parse(cx, &mbe_matcher, args) {
        Success(map) => map,
        Failure(_, tok) => {
            panic!("expected Success, but got Failure: {}", parse_failure_msg(tok));
        }
        Error(_, s) => {
            panic!("expected Success, but got Error: {}", s);
        }
    };

    let matched_nt = match *map[&Ident::from_str("matched")] {
        MatchedNonterminal(ref nt) => nt.clone(),
        _ => unreachable!(),
    };

    let mac_expr = match (&*matched_nt, &*map[&Ident::from_str("pat")]) {
        (&NtExpr(ref matched_expr), &MatchedSeq(ref pats, seq_sp)) => {
            let pats: Vec<P<Pat>> = pats.iter().map(|pat_nt| {
                match **pat_nt {
                    MatchedNonterminal(ref nt) => match **nt {
                        NtPat(ref pat) => pat.clone(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }).collect();
            let arm = cx.arm(seq_sp, pats, cx.expr_bool(seq_sp, true));

            quote_expr!(cx,
                match $matched_expr {
                    $arm
                    _ => false
                }
            )
        }
        _ => unreachable!()
    };

    MacEager::expr(mac_expr)
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("matches", expand_mbe_matches);
}
