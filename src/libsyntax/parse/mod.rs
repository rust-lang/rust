// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The main parser interface

use ast::node_id;
use ast;
use codemap::{span, CodeMap, FileMap, CharPos, BytePos};
use codemap;
use diagnostic::{span_handler, mk_span_handler, mk_handler, emitter};
use parse::attr::parser_attr;
use parse::lexer::{reader, string_reader};
use parse::parser::Parser;
use parse::token::{ident_interner, mk_ident_interner};
use util::interner;

use core::io;
use core::option::{None, Option, Some};
use core::path::Path;
use core::result::{Err, Ok, Result};
use core::result;

#[legacy_exports]
pub mod lexer;
#[legacy_exports]
pub mod parser;
#[legacy_exports]
pub mod token;
#[legacy_exports]
pub mod comments;
#[legacy_exports]
pub mod attr;
#[legacy_exports]

/// Common routines shared by parser mods
#[legacy_exports]
pub mod common;

/// Functions dealing with operator precedence
#[legacy_exports]
pub mod prec;

/// Routines the parser uses to classify AST nodes
#[legacy_exports]
pub mod classify;

/// Reporting obsolete syntax
#[legacy_exports]
pub mod obsolete;

pub type parse_sess = @{
    cm: @codemap::CodeMap,
    mut next_id: node_id,
    span_diagnostic: span_handler,
    interner: @ident_interner,
};

pub fn new_parse_sess(demitter: Option<emitter>) -> parse_sess {
    let cm = @CodeMap::new();
    return @{cm: cm,
             mut next_id: 1,
             span_diagnostic: mk_span_handler(mk_handler(demitter), cm),
             interner: mk_ident_interner(),
            };
}

pub fn new_parse_sess_special_handler(sh: span_handler, cm: @codemap::CodeMap)
    -> parse_sess {
    return @{cm: cm,
             mut next_id: 1,
             span_diagnostic: sh,
             interner: mk_ident_interner(),
             };
}

pub fn parse_crate_from_file(input: &Path, cfg: ast::crate_cfg,
                         sess: parse_sess) -> @ast::crate {
    let p = new_crate_parser_from_file(sess, cfg, input);
    let r = p.parse_crate_mod(cfg);
    return r;
}

pub fn parse_crate_from_source_str(name: ~str,
                                   source: @~str,
                                   cfg: ast::crate_cfg,
                                   sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_crate_mod(cfg);
    p.abort_if_errors();
    return r;
}

pub fn parse_expr_from_source_str(name: ~str,
                                  source: @~str,
                                  cfg: ast::crate_cfg,
                                  sess: parse_sess) -> @ast::expr {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_expr();
    p.abort_if_errors();
    return r;
}

pub fn parse_item_from_source_str(name: ~str,
                                  source: @~str,
                                  cfg: ast::crate_cfg,
                                  +attrs: ~[ast::attribute],
                                  sess: parse_sess)
                               -> Option<@ast::item> {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_item(attrs);
    p.abort_if_errors();
    return r;
}

pub fn parse_stmt_from_source_str(name: ~str,
                                  source: @~str,
                                  cfg: ast::crate_cfg,
                                  +attrs: ~[ast::attribute],
                                  sess: parse_sess) -> @ast::stmt {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_stmt(attrs);
    p.abort_if_errors();
    return r;
}

pub fn parse_tts_from_source_str(name: ~str,
                                 source: @~str,
                                 cfg: ast::crate_cfg,
                                 sess: parse_sess) -> ~[ast::token_tree] {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    p.quote_depth += 1u;
    let r = p.parse_all_token_trees();
    p.abort_if_errors();
    return r;
}

pub fn parse_from_source_str<T>(f: fn (p: Parser) -> T,
                            name: ~str, ss: codemap::FileSubstr,
                            source: @~str, cfg: ast::crate_cfg,
                            sess: parse_sess)
    -> T
{
    let p = new_parser_from_source_str(sess, cfg, name, ss,
                                       source);
    let r = f(p);
    if !p.reader.is_eof() {
        p.reader.fatal(~"expected end-of-string");
    }
    p.abort_if_errors();
    move r
}

pub fn next_node_id(sess: parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert rv != 0;
    return rv;
}

pub fn new_parser_from_source_str(sess: parse_sess, cfg: ast::crate_cfg,
                              +name: ~str, +ss: codemap::FileSubstr,
                              source: @~str) -> Parser {
    let filemap = sess.cm.new_filemap_w_substr(name, ss, source);
    let srdr = lexer::new_string_reader(sess.span_diagnostic, filemap,
                                        sess.interner);
    return Parser(sess, cfg, srdr as reader);
}

pub fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                        path: &Path) -> Result<Parser, ~str> {
    match io::read_whole_file_str(path) {
      result::Ok(move src) => {

          let filemap = sess.cm.new_filemap(path.to_str(), @move src);
          let srdr = lexer::new_string_reader(sess.span_diagnostic, filemap,
                                              sess.interner);

          Ok(Parser(sess, cfg, srdr as reader))

      }
      result::Err(move e) => Err(move e)
    }
}

/// Create a new parser for an entire crate, handling errors as appropriate
/// if the file doesn't exist
pub fn new_crate_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                              path: &Path) -> Parser {
    match new_parser_from_file(sess, cfg, path) {
        Ok(move parser) => move parser,
        Err(move e) => {
            sess.span_diagnostic.handler().fatal(e)
        }
    }
}

/// Create a new parser based on a span from an existing parser. Handles
/// error messages correctly when the file does not exist.
pub fn new_sub_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                            path: &Path, sp: span) -> Parser {
    match new_parser_from_file(sess, cfg, path) {
        Ok(move parser) => move parser,
        Err(move e) => {
            sess.span_diagnostic.span_fatal(sp, e)
        }
    }
}

pub fn new_parser_from_tts(sess: parse_sess, cfg: ast::crate_cfg,
                       tts: ~[ast::token_tree]) -> Parser {
    let trdr = lexer::new_tt_reader(sess.span_diagnostic, sess.interner,
                                    None, tts);
    return Parser(sess, cfg, trdr as reader)
}
