// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The token-tree map syntax extension maintains an arbitrary number
 * of named maps from ident to token-tree. Token trees can be stored
 * into a map with the `__tt_map_insert!` macro and retrieved as an
 * expression with the `__tt_map_get_expr!` macro.
 *
 * This is a hack used to maintain the tables of constants used by
 * the rustc error reporting system. In particular, it allows string
 * literals to be reused in multiple places without duplication.
 */

use ast;
use codemap::Span;
use ext::base::{ExtCtxt, MacResult, MRExpr, MRItem};
use ext::build::AstBuilder;
use parse::token;
use parse::token::{gensym, interner_get};
use parse::new_parser_from_tts;
use std::hashmap::HashMap;

pub fn insert_expr(ecx: &mut ExtCtxt, sp: Span,
                   tts: &[ast::TokenTree]) -> MacResult {

    if tts.len() != 5 {
        ecx.span_fatal(sp, "incorrect number of arguments");
    }

    let idxs = [1, 3];
    for i in idxs.iter() {
        match &tts[*i] {
            &ast::TTTok(_, token::COMMA) => (),
            _ => ecx.span_fatal(sp, "expecting comma")
        }
    }

    let map_name = tree_2_name(ecx, &tts[0]);
    let key_name = tree_2_name(ecx, &tts[2]);
    let expr = tts[4].clone();

    if !ecx.tt_maps.contains_key(&map_name) {
        ecx.tt_maps.insert(map_name.clone(), HashMap::new());
    }

    let existed = {
        let mut maybe_map = ecx.tt_maps.find_mut(&map_name);
        let map = maybe_map.get_mut_ref();
        !map.insert(key_name, expr)
    };

    if existed {
        let key_name = interner_get(key_name);
        ecx.span_fatal(sp, format!("key {} already exists in map", key_name));
    }

    // This item isn't used
    let dummy_ident = ast::Ident::new(gensym("dummy_name"));
    let dummy_item = ecx.item_mod(sp, dummy_ident, ~[], ~[], ~[]);
    return MRItem(dummy_item);
}

pub fn get_expr(ecx: &mut ExtCtxt, sp: Span,
                tts: &[ast::TokenTree]) -> MacResult {

    if tts.len() != 3 {
        ecx.span_fatal(sp, "incorrect number of arguments");
    }

    match &tts[1] {
        &ast::TTTok(_, token::COMMA) => (),
        _ => ecx.span_fatal(sp, "expecting comma")
    }

    let map_name = tree_2_name(ecx, &tts[0]);
    let key_name = tree_2_name(ecx, &tts[2]);

    match ecx.tt_maps.find(&map_name) {
        Some(map) => {
            match map.find(&key_name) {
                Some(map_tree) => {
                    MRExpr(tree_2_expr(ecx, map_tree))
                }
                None => {
                    let key_name = interner_get(key_name);
                    ecx.span_fatal(sp, format!("key {} does not exist in map", key_name));
                }
            }
        }
        None => {
            ecx.span_fatal(sp, "map does not exist");
        }
    }
}

fn tree_2_name(ecx: &ExtCtxt, tts: &ast::TokenTree) -> ast::Name {
    let mut p = new_parser_from_tts(ecx.parse_sess(), ecx.cfg.clone(), ~[tts.clone()]);
    return p.parse_ident().name;
}

fn tree_2_expr(ecx: &ExtCtxt, tts: &ast::TokenTree) -> @ast::Expr {
    let mut p = new_parser_from_tts(ecx.parse_sess(), ecx.cfg.clone(), ~[tts.clone()]);
    return p.parse_expr();
}
