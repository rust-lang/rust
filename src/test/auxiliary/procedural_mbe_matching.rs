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
extern crate rustc;

use syntax::codemap::Span;
use syntax::parse::token::{self, str_to_ident, NtExpr, NtPat};
use syntax::ast::{TokenTree, Pat};
use syntax::ext::base::{ExtCtxt, MacResult, DummyResult, MacEager};
use syntax::ext::build::AstBuilder;
use syntax::ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use syntax::ext::tt::macro_parser::{Success, Failure, Error};
use syntax::ptr::P;
use rustc::plugin::Registry;

fn expand_mbe_matches(cx: &mut ExtCtxt, sp: Span, args: &[TokenTree])
        -> Box<MacResult + 'static> {

    let mbe_matcher = quote_matcher!(cx, $matched:expr, $($pat:pat)|+);

    let mac_expr = match TokenTree::parse(cx, &mbe_matcher[..], args) {
        Success(map) => {
            match (&*map[&str_to_ident("matched").name], &*map[&str_to_ident("pat").name]) {
                (&MatchedNonterminal(NtExpr(ref matched_expr)),
                 &MatchedSeq(ref pats, seq_sp)) => {
                    let pats: Vec<P<Pat>> = pats.iter().map(|pat_nt|
                        if let &MatchedNonterminal(NtPat(ref pat)) = &**pat_nt {
                            pat.clone()
                        } else {
                            unreachable!()
                        }
                    ).collect();
                    let arm = cx.arm(seq_sp, pats, cx.expr_bool(seq_sp, true));

                    quote_expr!(cx,
                        match $matched_expr {
                            $arm
                            _ => false
                        }
                    )
                }
                _ => unreachable!()
            }
        }
        Failure(_, s) | Error(_, s) => {
            panic!("expected Success, but got Error/Failure: {}", s);
        }
    };

    MacEager::expr(mac_expr)
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("matches", expand_mbe_matches);
}
