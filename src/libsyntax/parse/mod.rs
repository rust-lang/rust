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
use codemap::{span, CodeMap, FileMap, FileSubstr};
use codemap;
use diagnostic::{span_handler, mk_span_handler, mk_handler, Emitter};
use parse::attr::parser_attr;
use parse::lexer::reader;
use parse::parser::Parser;

use core::io;
use core::option::{None, Option, Some};
use core::path::Path;
use core::result::{Err, Ok};

pub mod lexer;
pub mod parser;
pub mod token;
pub mod comments;
pub mod attr;


/// Common routines shared by parser mods
pub mod common;

/// Routines the parser uses to classify AST nodes
pub mod classify;

/// Reporting obsolete syntax
pub mod obsolete;

// info about a parsing session.
pub struct ParseSess {
    cm: @codemap::CodeMap, // better be the same as the one in the reader!
    next_id: node_id,
    span_diagnostic: @span_handler, // better be the same as the one in the reader!
}

pub fn new_parse_sess(demitter: Option<Emitter>) -> @mut ParseSess {
    let cm = @CodeMap::new();
    @mut ParseSess {
        cm: cm,
        next_id: 1,
        span_diagnostic: mk_span_handler(mk_handler(demitter), cm),
    }
}

pub fn new_parse_sess_special_handler(sh: @span_handler,
                                      cm: @codemap::CodeMap)
                                   -> @mut ParseSess {
    @mut ParseSess {
        cm: cm,
        next_id: 1,
        span_diagnostic: sh,
    }
}

// a bunch of utility functions of the form parse_<thing>_from_<source>
// where <thing> includes crate, expr, item, stmt, tts, and one that
// uses a HOF to parse anything, and <source> includes file and
// source_str.

pub fn parse_crate_from_file(
    input: &Path,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> @ast::crate {
    new_parser_from_file(sess, /*bad*/ copy cfg, input).parse_crate_mod()
    // why is there no p.abort_if_errors here?
}

pub fn parse_crate_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> @ast::crate {
    let p = new_parser_from_source_str(
        sess,
        /*bad*/ copy cfg,
        /*bad*/ copy name,
        source
    );
    maybe_aborted(p.parse_crate_mod(),p)
}

pub fn parse_expr_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> @ast::expr {
    let p = new_parser_from_source_str(
        sess,
        cfg,
        /*bad*/ copy name,
        source
    );
    maybe_aborted(p.parse_expr(), p)
}

pub fn parse_item_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    attrs: ~[ast::attribute],
    sess: @mut ParseSess
) -> Option<@ast::item> {
    let p = new_parser_from_source_str(
        sess,
        cfg,
        /*bad*/ copy name,
        source
    );
    maybe_aborted(p.parse_item(attrs),p)
}

pub fn parse_meta_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> @ast::meta_item {
    let p = new_parser_from_source_str(
        sess,
        cfg,
        /*bad*/ copy name,
        source
    );
    maybe_aborted(p.parse_meta_item(),p)
}

pub fn parse_stmt_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    attrs: ~[ast::attribute],
    sess: @mut ParseSess
) -> @ast::stmt {
    let p = new_parser_from_source_str(
        sess,
        cfg,
        /*bad*/ copy name,
        source
    );
    maybe_aborted(p.parse_stmt(attrs),p)
}

pub fn parse_tts_from_source_str(
    name: ~str,
    source: @~str,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> ~[ast::token_tree] {
    let p = new_parser_from_source_str(
        sess,
        cfg,
        /*bad*/ copy name,
        source
    );
    *p.quote_depth += 1u;
    // right now this is re-creating the token trees from ... token trees.
    maybe_aborted(p.parse_all_token_trees(),p)
}

// given a function and parsing information (source str,
// filename, crate cfg, and sess), create a parser,
// apply the function, and check that the parser
// consumed all of the input before returning the function's
// result.
pub fn parse_from_source_str<T>(
    f: &fn(&Parser) -> T,
    name: ~str, ss: codemap::FileSubstr,
    source: @~str,
    cfg: ast::crate_cfg,
    sess: @mut ParseSess
) -> T {
    let p = new_parser_from_source_substr(
        sess,
        cfg,
        name,
        ss,
        source
    );
    let r = f(&p);
    if !p.reader.is_eof() {
        p.reader.fatal(~"expected end-of-string");
    }
    maybe_aborted(r,p)
}

// return the next unused node id.
pub fn next_node_id(sess: @mut ParseSess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert!(rv != 0);
    return rv;
}

// Create a new parser from a source string
pub fn new_parser_from_source_str(sess: @mut ParseSess,
                                  cfg: ast::crate_cfg,
                                  name: ~str,
                                  source: @~str)
                               -> Parser {
    filemap_to_parser(sess,string_to_filemap(sess,source,name),cfg)
}

// Create a new parser from a source string where the origin
// is specified as a substring of another file.
pub fn new_parser_from_source_substr(sess: @mut ParseSess,
                                  cfg: ast::crate_cfg,
                                  name: ~str,
                                  ss: codemap::FileSubstr,
                                  source: @~str)
                               -> Parser {
    filemap_to_parser(sess,substring_to_filemap(sess,source,name,ss),cfg)
}

/// Create a new parser, handling errors as appropriate
/// if the file doesn't exist
pub fn new_parser_from_file(
    sess: @mut ParseSess,
    cfg: ast::crate_cfg,
    path: &Path
) -> Parser {
    filemap_to_parser(sess,file_to_filemap(sess,path,None),cfg)
}

/// Given a session, a crate config, a path, and a span, add
/// the file at the given path to the codemap, and return a parser.
/// On an error, use the given span as the source of the problem.
pub fn new_sub_parser_from_file(
    sess: @mut ParseSess,
    cfg: ast::crate_cfg,
    path: &Path,
    sp: span
) -> Parser {
    filemap_to_parser(sess,file_to_filemap(sess,path,Some(sp)),cfg)
}

/// Given a filemap and config, return a parser
pub fn filemap_to_parser(sess: @mut ParseSess,
                         filemap: @FileMap,
                         cfg: ast::crate_cfg) -> Parser {
    tts_to_parser(sess,filemap_to_tts(sess,filemap),cfg)
}

// must preserve old name for now, because quote! from the *existing*
// compiler expands into it
pub fn new_parser_from_tts(sess: @mut ParseSess,
                     cfg: ast::crate_cfg,
                     tts: ~[ast::token_tree]) -> Parser {
    tts_to_parser(sess,tts,cfg)
}


// base abstractions

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's codemap and return the new filemap.
pub fn file_to_filemap(sess: @mut ParseSess, path: &Path, spanopt: Option<span>)
    -> @FileMap {
    match io::read_whole_file_str(path) {
        Ok(src) => string_to_filemap(sess, @src, path.to_str()),
        Err(e) => {
            match spanopt {
                Some(span) => sess.span_diagnostic.span_fatal(span, e),
                None => sess.span_diagnostic.handler().fatal(e)
            }
        }
    }
}

// given a session and a string, add the string to
// the session's codemap and return the new filemap
pub fn string_to_filemap(sess: @mut ParseSess, source: @~str, path: ~str)
    -> @FileMap {
    sess.cm.new_filemap(path, source)
}

// given a session and a string and a path and a FileSubStr, add
// the string to the CodeMap and return the new FileMap
pub fn substring_to_filemap(sess: @mut ParseSess, source: @~str, path: ~str,
                           filesubstr: FileSubstr) -> @FileMap {
    sess.cm.new_filemap_w_substr(path,filesubstr,source)
}

// given a filemap, produce a sequence of token-trees
pub fn filemap_to_tts(sess: @mut ParseSess, filemap: @FileMap)
    -> ~[ast::token_tree] {
    // it appears to me that the cfg doesn't matter here... indeed,
    // parsing tt's probably shouldn't require a parser at all.
    let cfg = ~[];
    let srdr = lexer::new_string_reader(copy sess.span_diagnostic, filemap);
    let p1 = Parser(sess, cfg, srdr as @reader);
    p1.parse_all_token_trees()
}

// given tts and cfg, produce a parser
pub fn tts_to_parser(sess: @mut ParseSess,
                     tts: ~[ast::token_tree],
                     cfg: ast::crate_cfg) -> Parser {
    let trdr = lexer::new_tt_reader(
        copy sess.span_diagnostic,
        None,
        tts
    );
    Parser(sess, cfg, trdr as @reader)
}

// abort if necessary
pub fn maybe_aborted<T>(result : T, p: Parser) -> T {
    p.abort_if_errors();
    result
}



#[cfg(test)]
mod test {
    use super::*;
    use extra::serialize::Encodable;
    use extra;
    use core::io;
    use core::option::Some;
    use core::option::None;
    use codemap::{span, BytePos, spanned};
    use opt_vec;
    use ast;
    use ast::{new_ident};
    use abi;
    use parse::parser::Parser;
    use parse::token::{intern, str_to_ident};
    use util::parser_testing::{string_to_tts_and_sess, string_to_parser};
    use util::parser_testing::{string_to_expr, string_to_item};
    use util::parser_testing::{string_to_stmt, strs_to_idents};

    // map a string to tts, return the tt without its parsesess
    fn string_to_tts_only(source_str : @~str) -> ~[ast::token_tree] {
        let (tts,_ps) = string_to_tts_and_sess(source_str);
        tts
    }


    #[cfg(test)] fn to_json_str<E : Encodable<extra::json::Encoder>>(val: @E) -> ~str {
        do io::with_str_writer |writer| {
            let mut encoder = extra::json::Encoder(writer);
            val.encode(&mut encoder);
        }
    }

    // produce a codemap::span
    fn sp (a: uint, b: uint) -> span {
        span{lo:BytePos(a),hi:BytePos(b),expn_info:None}
    }

    #[test] fn path_exprs_1 () {
        assert_eq!(string_to_expr(@~"a"),
                   @ast::expr{id:1,
                              node:ast::expr_path(@ast::Path {span:sp(0,1),
                                                              global:false,
                                                              idents:~[str_to_ident("a")],
                                                              rp:None,
                                                              types:~[]}),
                              span:sp(0,1)})
    }

    #[test] fn path_exprs_2 () {
        assert_eq!(string_to_expr(@~"::a::b"),
                   @ast::expr{id:1,
                               node:ast::expr_path(
                                   @ast::Path {span:sp(0,6),
                                               global:true,
                                               idents:strs_to_idents(~["a","b"]),
                                               rp:None,
                                               types:~[]}),
                              span:sp(0,6)})
    }

    // FIXME (#6416): For some reason, this fails and causes a test failure, even though it's
    // marked as `#[should_fail]`.
    /*#[should_fail]
    #[test] fn bad_path_expr_1() {
        string_to_expr(@~"::abc::def::return");
    }*/

    #[test] fn string_to_tts_1 () {
        let (tts,_ps) = string_to_tts_and_sess(@~"fn a (b : int) { b; }");
        assert_eq!(to_json_str(@tts),
                   ~"[\
                [\"tt_tok\",null,[\"IDENT\",\"fn\",false]],\
                [\"tt_tok\",null,[\"IDENT\",\"a\",false]],\
                [\
                    \"tt_delim\",\
                    [\
                        [\"tt_tok\",null,\"LPAREN\"],\
                        [\"tt_tok\",null,[\"IDENT\",\"b\",false]],\
                        [\"tt_tok\",null,\"COLON\"],\
                        [\"tt_tok\",null,[\"IDENT\",\"int\",false]],\
                        [\"tt_tok\",null,\"RPAREN\"]\
                    ]\
                ],\
                [\
                    \"tt_delim\",\
                    [\
                        [\"tt_tok\",null,\"LBRACE\"],\
                        [\"tt_tok\",null,[\"IDENT\",\"b\",false]],\
                        [\"tt_tok\",null,\"SEMI\"],\
                        [\"tt_tok\",null,\"RBRACE\"]\
                    ]\
                ]\
            ]"
                  );
    }

    #[test] fn ret_expr() {
        assert_eq!(string_to_expr(@~"return d"),
                   @ast::expr{id:2,
                              node:ast::expr_ret(
                                  Some(@ast::expr{id:1,
                                                  node:ast::expr_path(
                                                      @ast::Path{span:sp(7,8),
                                                                 global:false,
                                                                 idents:~[str_to_ident("d")],
                                                                 rp:None,
                                                                 types:~[]
                                                                }),
                                                  span:sp(7,8)})),
                              span:sp(0,8)})
    }

    #[test] fn parse_stmt_1 () {
        assert_eq!(string_to_stmt(@~"b;"),
                   @spanned{
                       node: ast::stmt_expr(@ast::expr{
                           id: 1,
                           node: ast::expr_path(
                               @ast::Path{
                                   span:sp(0,1),
                                   global:false,
                                   idents:~[str_to_ident("b")],
                                   rp:None,
                                   types: ~[]}),
                           span: sp(0,1)},
                                            2), // fixme
                       span: sp(0,1)})

    }

    fn parser_done(p: Parser){
        assert_eq!(copy *p.token,token::EOF);
    }

    #[test] fn parse_ident_pat () {
        let parser = string_to_parser(@~"b");
        assert_eq!(parser.parse_pat(),
                   @ast::pat{id:1, // fixme
                             node: ast::pat_ident(ast::bind_infer,
                                                  @ast::Path{
                                                      span:sp(0,1),
                                                      global:false,
                                                      idents:~[str_to_ident("b")],
                                                      rp: None,
                                                      types: ~[]},
                                                  None // no idea
                                                 ),
                             span: sp(0,1)});
        parser_done(parser);
    }

    #[test] fn parse_arg () {
        let parser = string_to_parser(@~"b : int");
        assert_eq!(parser.parse_arg_general(true),
                   ast::arg{
                       is_mutbl: false,
                       ty: @ast::Ty{id:3, // fixme
                                    node: ast::ty_path(@ast::Path{
                                        span:sp(4,4), // this is bizarre...
                                        // check this in the original parser?
                                        global:false,
                                        idents:~[str_to_ident("int")],
                                        rp: None,
                                        types: ~[]},
                                                       2),
                                    span:sp(4,7)},
                       pat: @ast::pat{id:1,
                                      node: ast::pat_ident(ast::bind_infer,
                                                           @ast::Path{
                                                               span:sp(0,1),
                                                               global:false,
                                                               idents:~[str_to_ident("b")],
                                                               rp: None,
                                                               types: ~[]},
                                                           None // no idea
                                                          ),
                                      span: sp(0,1)},
                       id: 4 // fixme
                   })
    }

    // check the contents of the tt manually:
    #[test] fn parse_fundecl () {
        // this test depends on the intern order of "fn" and "int", and on the
        // assignment order of the node_ids.
        assert_eq!(string_to_item(@~"fn a (b : int) { b; }"),
                  Some(
                      @ast::item{ident:str_to_ident("a"),
                            attrs:~[],
                            id: 9, // fixme
                            node: ast::item_fn(ast::fn_decl{
                                inputs: ~[ast::arg{
                                    is_mutbl: false,
                                    ty: @ast::Ty{id:3, // fixme
                                                node: ast::ty_path(@ast::Path{
                                        span:sp(10,13),
                                        global:false,
                                        idents:~[str_to_ident("int")],
                                        rp: None,
                                        types: ~[]},
                                                       2),
                                                span:sp(10,13)},
                                    pat: @ast::pat{id:1, // fixme
                                                   node: ast::pat_ident(
                                                       ast::bind_infer,
                                                       @ast::Path{
                                                           span:sp(6,7),
                                                           global:false,
                                                           idents:~[str_to_ident("b")],
                                                           rp: None,
                                                           types: ~[]},
                                                       None // no idea
                                                   ),
                                                  span: sp(6,7)},
                                    id: 4 // fixme
                                }],
                                output: @ast::Ty{id:5, // fixme
                                                 node: ast::ty_nil,
                                                 span:sp(15,15)}, // not sure
                                cf: ast::return_val
                            },
                                    ast::impure_fn,
                                    abi::AbiSet::Rust(),
                                    ast::Generics{ // no idea on either of these:
                                        lifetimes: opt_vec::Empty,
                                        ty_params: opt_vec::Empty,
                                    },
                                    spanned{
                                        span: sp(15,21),
                                        node: ast::blk_{
                                            view_items: ~[],
                                            stmts: ~[@spanned{
                                                node: ast::stmt_semi(@ast::expr{
                                                    id: 6,
                                                    node: ast::expr_path(
                                                        @ast::Path{
                                                            span:sp(17,18),
                                                            global:false,
                                                            idents:~[str_to_ident("b")],
                                                            rp:None,
                                                            types: ~[]}),
                                                    span: sp(17,18)},
                                                                     7), // fixme
                                                span: sp(17,18)}],
                                            expr: None,
                                            id: 8, // fixme
                                            rules: ast::default_blk // no idea
                                        }}),
                            vis: ast::inherited,
                            span: sp(0,21)}));
    }


    #[test] fn parse_exprs () {
        // just make sure that they parse....
        string_to_expr(@~"3 + 4");
        string_to_expr(@~"a::z.froob(b,@(987+3))");
    }

    #[test] fn attrs_fix_bug () {
        string_to_item(@~"pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<@Writer, ~str> {
    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
}");
    }

}
