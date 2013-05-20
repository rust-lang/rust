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
use ast;
use parse::parser::Parser;
use parse::{new_parse_sess};

use syntax::parse::{ParseSess,string_to_filemap,filemap_to_tts};
use syntax::parse::{new_parser_from_source_str};

// map a string to tts, using a made-up filename: return both the token_trees
// and the ParseSess
pub fn string_to_tts_and_sess (source_str : @~str) -> (~[ast::token_tree],@mut ParseSess) {
    let ps = new_parse_sess(None);
    (filemap_to_tts(ps,string_to_filemap(ps,source_str,~"bogofile")),ps)
}

pub fn string_to_parser_and_sess(source_str: @~str) -> (Parser,@mut ParseSess) {
    let ps = new_parse_sess(None);
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

