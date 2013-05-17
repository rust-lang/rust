// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::option::{Option,None};
use core::int;
use core::num::NumCast;
use codemap::CodeMap;
use ast;
use parse::parser::Parser;
use parse::token::{ident_interner, mk_fresh_ident_interner};
use diagnostic::{mk_handler, mk_span_handler};

use syntax::parse::{ParseSess,string_to_filemap,filemap_to_tts};
use syntax::parse::{new_parser_from_source_str};

// add known names to interner for testing
fn mk_testing_interner() -> @ident_interner {
    let i = mk_fresh_ident_interner();
    // baby hack; in order to put the identifiers
    // 'a' and 'b' at known locations, we're going
    // to fill up the interner to length 100. If
    // the # of preloaded items on the interner
    // ever gets larger than 100, we'll have to
    // adjust this number (say, to 200) and
    // change the numbers in the identifier
    // test cases below.

    assert!(i.len() < 100);
    for int::range(0,100-((i.len()).to_int())) |_dc| {
        i.gensym(~"dontcare");
    }
    i.intern("a");
    i.intern("b");
    i.intern("c");
    i.intern("d");
    i.intern("return");
    assert!(i.get(ast::ident{repr:101,ctxt:0}) == @~"b");
    i
}

// make a parse_sess that's closed over a
// testing interner (where a -> 100, b -> 101)
fn mk_testing_parse_sess() -> @mut ParseSess {
    let interner = mk_testing_interner();
    let cm = @CodeMap::new();
    @mut ParseSess {
        cm: cm,
        next_id: 1,
        span_diagnostic: mk_span_handler(mk_handler(None), cm),
        interner: interner,
    }
}

// map a string to tts, using a made-up filename: return both the token_trees
// and the ParseSess
pub fn string_to_tts_and_sess (source_str : @~str) -> (~[ast::token_tree],@mut ParseSess) {
    let ps = mk_testing_parse_sess();
    (filemap_to_tts(ps,string_to_filemap(ps,source_str,~"bogofile")),ps)
}

pub fn string_to_parser_and_sess(source_str: @~str) -> (Parser,@mut ParseSess) {
    let ps = mk_testing_parse_sess();
    (new_parser_from_source_str(ps,~[],~"bogofile",source_str),ps)
}

// map string to parser (via tts)
pub fn string_to_parser(source_str: @~str) -> Parser {
    let (p,_) = string_to_parser_and_sess(source_str);
    p
}

pub fn string_to_crate (source_str : @~str) -> @ast::crate {
    string_to_parser(source_str).parse_crate_mod()
}

// parse a string, return an expr
pub fn string_to_expr (source_str : @~str) -> @ast::expr {
    string_to_parser(source_str).parse_expr()
}

// parse a string, return an item
pub fn string_to_item (source_str : @~str) -> Option<@ast::item> {
    string_to_parser(source_str).parse_item(~[])
}

// parse a string, return an item and the ParseSess
pub fn string_to_item_and_sess (source_str : @~str) -> (Option<@ast::item>,@mut ParseSess) {
    let (p,ps) = string_to_parser_and_sess(source_str);
    (p.parse_item(~[]),ps)
}

pub fn string_to_stmt (source_str : @~str) -> @ast::stmt {
    string_to_parser(source_str).parse_stmt(~[])
}

