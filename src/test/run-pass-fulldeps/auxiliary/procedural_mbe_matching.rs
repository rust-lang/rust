// force-host

#![crate_type="dylib"]
#![feature(plugin_registrar, quote, rustc_private)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc;
extern crate rustc_plugin;

use syntax::feature_gate::Features;
use syntax::parse::token::{NtExpr, NtPat};
use syntax::ast::{Ident, Pat, NodeId};
use syntax::tokenstream::{TokenTree};
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::ext::build::AstBuilder;
use syntax::ext::tt::quoted;
use syntax::ext::tt::macro_parser::{MatchedSeq, MatchedNonterminal};
use syntax::ext::tt::macro_parser::{Success, Failure, Error};
use syntax::ext::tt::macro_parser::parse_failure_msg;
use syntax::ptr::P;
use syntax_pos::{Span, edition::Edition};
use rustc_plugin::Registry;

fn expand_mbe_matches(cx: &mut ExtCtxt, _: Span, args: &[TokenTree])
        -> Box<MacResult + 'static> {

    let mbe_matcher = quote_tokens!(cx, $$matched:expr, $$($$pat:pat)|+);
    let mbe_matcher = quoted::parse(mbe_matcher.into_iter().collect(),
                                    true,
                                    cx.parse_sess,
                                    &Features::new(),
                                    &[],
                                    Edition::Edition2015,
                                    // not used...
                                    NodeId::from_u32(0));
    let map = match TokenTree::parse(cx, &mbe_matcher, args.iter().cloned().collect()) {
        Success(map) => map,
        Failure(_, tok, msg) => {
            panic!("expected Success, but got Failure: {} - {}", parse_failure_msg(tok), msg);
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
                match *pat_nt {
                    MatchedNonterminal(ref nt) => match **nt {
                        NtPat(ref pat) => pat.clone(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }).collect();
            let span = seq_sp.entire();
            let arm = cx.arm(span, pats, cx.expr_bool(span, true));

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
