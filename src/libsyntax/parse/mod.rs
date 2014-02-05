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


use ast;
use codemap::{Span, CodeMap, FileMap};
use codemap;
use diagnostic::{SpanHandler, mk_span_handler, mk_handler, Emitter};
use parse::attr::ParserAttr;
use parse::parser::Parser;

use std::cell::RefCell;
use std::io::File;
use std::str;

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
    span_diagnostic: @SpanHandler, // better be the same as the one in the reader!
    /// Used to determine and report recursive mod inclusions
    included_mod_stack: RefCell<~[Path]>,
}

pub fn new_parse_sess(demitter: Option<@Emitter>) -> @ParseSess {
    let cm = @CodeMap::new();
    @ParseSess {
        cm: cm,
        span_diagnostic: mk_span_handler(mk_handler(demitter), cm),
        included_mod_stack: RefCell::new(~[]),
    }
}

pub fn new_parse_sess_special_handler(sh: @SpanHandler,
                                      cm: @codemap::CodeMap)
                                      -> @ParseSess {
    @ParseSess {
        cm: cm,
        span_diagnostic: sh,
        included_mod_stack: RefCell::new(~[]),
    }
}

// a bunch of utility functions of the form parse_<thing>_from_<source>
// where <thing> includes crate, expr, item, stmt, tts, and one that
// uses a HOF to parse anything, and <source> includes file and
// source_str.

pub fn parse_crate_from_file(
    input: &Path,
    cfg: ast::CrateConfig,
    sess: @ParseSess
) -> ast::Crate {
    new_parser_from_file(sess, /*bad*/ cfg.clone(), input).parse_crate_mod()
    // why is there no p.abort_if_errors here?
}

pub fn parse_crate_attrs_from_file(
    input: &Path,
    cfg: ast::CrateConfig,
    sess: @ParseSess
) -> ~[ast::Attribute] {
    let mut parser = new_parser_from_file(sess, cfg, input);
    let (inner, _) = parser.parse_inner_attrs_and_next();
    return inner;
}

pub fn parse_crate_from_source_str(name: ~str,
                                   source: ~str,
                                   cfg: ast::CrateConfig,
                                   sess: @ParseSess)
                                   -> ast::Crate {
    let mut p = new_parser_from_source_str(sess,
                                           /*bad*/ cfg.clone(),
                                           name,
                                           source);
    maybe_aborted(p.parse_crate_mod(),p)
}

pub fn parse_crate_attrs_from_source_str(name: ~str,
                                         source: ~str,
                                         cfg: ast::CrateConfig,
                                         sess: @ParseSess)
                                         -> ~[ast::Attribute] {
    let mut p = new_parser_from_source_str(sess,
                                           /*bad*/ cfg.clone(),
                                           name,
                                           source);
    let (inner, _) = maybe_aborted(p.parse_inner_attrs_and_next(),p);
    return inner;
}

pub fn parse_expr_from_source_str(name: ~str,
                                  source: ~str,
                                  cfg: ast::CrateConfig,
                                  sess: @ParseSess)
                                  -> @ast::Expr {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_expr(), p)
}

pub fn parse_item_from_source_str(name: ~str,
                                  source: ~str,
                                  cfg: ast::CrateConfig,
                                  sess: @ParseSess)
                                  -> Option<@ast::Item> {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    let attrs = p.parse_outer_attributes();
    maybe_aborted(p.parse_item(attrs),p)
}

pub fn parse_meta_from_source_str(name: ~str,
                                  source: ~str,
                                  cfg: ast::CrateConfig,
                                  sess: @ParseSess)
                                  -> @ast::MetaItem {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_meta_item(),p)
}

pub fn parse_stmt_from_source_str(name: ~str,
                                  source: ~str,
                                  cfg: ast::CrateConfig,
                                  attrs: ~[ast::Attribute],
                                  sess: @ParseSess)
                                  -> @ast::Stmt {
    let mut p = new_parser_from_source_str(
        sess,
        cfg,
        name,
        source
    );
    maybe_aborted(p.parse_stmt(attrs),p)
}

pub fn parse_tts_from_source_str(name: ~str,
                                 source: ~str,
                                 cfg: ast::CrateConfig,
                                 sess: @ParseSess)
                                 -> ~[ast::TokenTree] {
    let mut p = new_parser_from_source_str(
        sess,
        cfg,
        name,
        source
    );
    p.quote_depth += 1u;
    // right now this is re-creating the token trees from ... token trees.
    maybe_aborted(p.parse_all_token_trees(),p)
}

// Create a new parser from a source string
pub fn new_parser_from_source_str(sess: @ParseSess,
                                  cfg: ast::CrateConfig,
                                  name: ~str,
                                  source: ~str)
                                  -> Parser {
    filemap_to_parser(sess,string_to_filemap(sess,source,name),cfg)
}

/// Create a new parser, handling errors as appropriate
/// if the file doesn't exist
pub fn new_parser_from_file(
    sess: @ParseSess,
    cfg: ast::CrateConfig,
    path: &Path
) -> Parser {
    filemap_to_parser(sess,file_to_filemap(sess,path,None),cfg)
}

/// Given a session, a crate config, a path, and a span, add
/// the file at the given path to the codemap, and return a parser.
/// On an error, use the given span as the source of the problem.
pub fn new_sub_parser_from_file(
    sess: @ParseSess,
    cfg: ast::CrateConfig,
    path: &Path,
    sp: Span
) -> Parser {
    filemap_to_parser(sess,file_to_filemap(sess,path,Some(sp)),cfg)
}

/// Given a filemap and config, return a parser
pub fn filemap_to_parser(sess: @ParseSess,
                         filemap: @FileMap,
                         cfg: ast::CrateConfig) -> Parser {
    tts_to_parser(sess,filemap_to_tts(sess,filemap),cfg)
}

// must preserve old name for now, because quote! from the *existing*
// compiler expands into it
pub fn new_parser_from_tts(sess: @ParseSess,
                     cfg: ast::CrateConfig,
                     tts: ~[ast::TokenTree]) -> Parser {
    tts_to_parser(sess,tts,cfg)
}


// base abstractions

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's codemap and return the new filemap.
pub fn file_to_filemap(sess: @ParseSess, path: &Path, spanopt: Option<Span>)
    -> @FileMap {
    let err = |msg: &str| {
        match spanopt {
            Some(sp) => sess.span_diagnostic.span_fatal(sp, msg),
            None => sess.span_diagnostic.handler().fatal(msg),
        }
    };
    let bytes = match File::open(path).read_to_end() {
        Ok(bytes) => bytes,
        Err(e) => {
            err(format!("couldn't read {}: {}", path.display(), e));
            unreachable!()
        }
    };
    match str::from_utf8_owned(bytes) {
        Some(s) => {
            return string_to_filemap(sess, s, path.as_str().unwrap().to_str())
        }
        None => err(format!("{} is not UTF-8 encoded", path.display())),
    }
    unreachable!()
}

// given a session and a string, add the string to
// the session's codemap and return the new filemap
pub fn string_to_filemap(sess: @ParseSess, source: ~str, path: ~str)
                         -> @FileMap {
    sess.cm.new_filemap(path, source)
}

// given a filemap, produce a sequence of token-trees
pub fn filemap_to_tts(sess: @ParseSess, filemap: @FileMap)
    -> ~[ast::TokenTree] {
    // it appears to me that the cfg doesn't matter here... indeed,
    // parsing tt's probably shouldn't require a parser at all.
    let cfg = ~[];
    let srdr = lexer::new_string_reader(sess.span_diagnostic, filemap);
    let mut p1 = Parser(sess, cfg, srdr as @lexer::Reader);
    p1.parse_all_token_trees()
}

// given tts and cfg, produce a parser
pub fn tts_to_parser(sess: @ParseSess,
                     tts: ~[ast::TokenTree],
                     cfg: ast::CrateConfig) -> Parser {
    let trdr = lexer::new_tt_reader(sess.span_diagnostic, None, tts);
    Parser(sess, cfg, trdr as @lexer::Reader)
}

// abort if necessary
pub fn maybe_aborted<T>(result: T, mut p: Parser) -> T {
    p.abort_if_errors();
    result
}



#[cfg(test)]
mod test {
    use super::*;
    use serialize::Encodable;
    use extra;
    use std::io;
    use std::io::MemWriter;
    use std::str;
    use codemap::{Span, BytePos, Spanned};
    use opt_vec;
    use ast;
    use abi;
    use parse::parser::Parser;
    use parse::token::{str_to_ident};
    use util::parser_testing::{string_to_tts, string_to_parser};
    use util::parser_testing::{string_to_expr, string_to_item};
    use util::parser_testing::string_to_stmt;

    #[cfg(test)]
    fn to_json_str<'a, E: Encodable<extra::json::Encoder<'a>>>(val: &E) -> ~str {
        let mut writer = MemWriter::new();
        let mut encoder = extra::json::Encoder::new(&mut writer as &mut io::Writer);
        val.encode(&mut encoder);
        str::from_utf8_owned(writer.unwrap()).unwrap()
    }

    // produce a codemap::span
    fn sp(a: u32, b: u32) -> Span {
        Span{lo:BytePos(a),hi:BytePos(b),expn_info:None}
    }

    #[test] fn path_exprs_1() {
        assert_eq!(string_to_expr(~"a"),
                   @ast::Expr{
                    id: ast::DUMMY_NODE_ID,
                    node: ast::ExprPath(ast::Path {
                        span: sp(0, 1),
                        global: false,
                        segments: ~[
                            ast::PathSegment {
                                identifier: str_to_ident("a"),
                                lifetimes: opt_vec::Empty,
                                types: opt_vec::Empty,
                            }
                        ],
                    }),
                    span: sp(0, 1)
                   })
    }

    #[test] fn path_exprs_2 () {
        assert_eq!(string_to_expr(~"::a::b"),
                   @ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    node: ast::ExprPath(ast::Path {
                            span: sp(0, 6),
                            global: true,
                            segments: ~[
                                ast::PathSegment {
                                    identifier: str_to_ident("a"),
                                    lifetimes: opt_vec::Empty,
                                    types: opt_vec::Empty,
                                },
                                ast::PathSegment {
                                    identifier: str_to_ident("b"),
                                    lifetimes: opt_vec::Empty,
                                    types: opt_vec::Empty,
                                }
                            ]
                        }),
                    span: sp(0, 6)
                   })
    }

    #[should_fail]
    #[test] fn bad_path_expr_1() {
        string_to_expr(~"::abc::def::return");
    }

    // check the token-tree-ization of macros
    #[test] fn string_to_tts_macro () {
        let tts = string_to_tts(~"macro_rules! zip (($a)=>($a))");
        match tts {
            [ast::TTTok(_,_),
             ast::TTTok(_,token::NOT),
             ast::TTTok(_,_),
             ast::TTDelim(delim_elts)] =>
                match *delim_elts {
                [ast::TTTok(_,token::LPAREN),
                 ast::TTDelim(first_set),
                 ast::TTTok(_,token::FAT_ARROW),
                 ast::TTDelim(second_set),
                 ast::TTTok(_,token::RPAREN)] =>
                    match *first_set {
                    [ast::TTTok(_,token::LPAREN),
                     ast::TTTok(_,token::DOLLAR),
                     ast::TTTok(_,_),
                     ast::TTTok(_,token::RPAREN)] =>
                        match *second_set {
                        [ast::TTTok(_,token::LPAREN),
                         ast::TTTok(_,token::DOLLAR),
                         ast::TTTok(_,_),
                         ast::TTTok(_,token::RPAREN)] =>
                            assert_eq!("correct","correct"),
                        _ => assert_eq!("wrong 4","correct")
                    },
                    _ => {
                        error!("failing value 3: {:?}",first_set);
                        assert_eq!("wrong 3","correct")
                    }
                },
                _ => {
                    error!("failing value 2: {:?}",delim_elts);
                    assert_eq!("wrong","correct");
                }

            },
            _ => {
                error!("failing value: {:?}",tts);
                assert_eq!("wrong 1","correct");
            }
        }
    }

    #[test] fn string_to_tts_1 () {
        let tts = string_to_tts(~"fn a (b : int) { b; }");
        assert_eq!(to_json_str(&tts),
        ~"[\
    {\
        \"variant\":\"TTTok\",\
        \"fields\":[\
            null,\
            {\
                \"variant\":\"IDENT\",\
                \"fields\":[\
                    \"fn\",\
                    false\
                ]\
            }\
        ]\
    },\
    {\
        \"variant\":\"TTTok\",\
        \"fields\":[\
            null,\
            {\
                \"variant\":\"IDENT\",\
                \"fields\":[\
                    \"a\",\
                    false\
                ]\
            }\
        ]\
    },\
    {\
        \"variant\":\"TTDelim\",\
        \"fields\":[\
            [\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"LPAREN\"\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        {\
                            \"variant\":\"IDENT\",\
                            \"fields\":[\
                                \"b\",\
                                false\
                            ]\
                        }\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"COLON\"\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        {\
                            \"variant\":\"IDENT\",\
                            \"fields\":[\
                                \"int\",\
                                false\
                            ]\
                        }\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"RPAREN\"\
                    ]\
                }\
            ]\
        ]\
    },\
    {\
        \"variant\":\"TTDelim\",\
        \"fields\":[\
            [\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"LBRACE\"\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        {\
                            \"variant\":\"IDENT\",\
                            \"fields\":[\
                                \"b\",\
                                false\
                            ]\
                        }\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"SEMI\"\
                    ]\
                },\
                {\
                    \"variant\":\"TTTok\",\
                    \"fields\":[\
                        null,\
                        \"RBRACE\"\
                    ]\
                }\
            ]\
        ]\
    }\
]"
        );
    }

    #[test] fn ret_expr() {
        assert_eq!(string_to_expr(~"return d"),
                   @ast::Expr{
                    id: ast::DUMMY_NODE_ID,
                    node:ast::ExprRet(Some(@ast::Expr{
                        id: ast::DUMMY_NODE_ID,
                        node:ast::ExprPath(ast::Path{
                            span: sp(7, 8),
                            global: false,
                            segments: ~[
                                ast::PathSegment {
                                    identifier: str_to_ident("d"),
                                    lifetimes: opt_vec::Empty,
                                    types: opt_vec::Empty,
                                }
                            ],
                        }),
                        span:sp(7,8)
                    })),
                    span:sp(0,8)
                   })
    }

    #[test] fn parse_stmt_1 () {
        assert_eq!(string_to_stmt(~"b;"),
                   @Spanned{
                       node: ast::StmtExpr(@ast::Expr {
                           id: ast::DUMMY_NODE_ID,
                           node: ast::ExprPath(ast::Path {
                               span:sp(0,1),
                               global:false,
                               segments: ~[
                                ast::PathSegment {
                                    identifier: str_to_ident("b"),
                                    lifetimes: opt_vec::Empty,
                                    types: opt_vec::Empty,
                                }
                               ],
                            }),
                           span: sp(0,1)},
                                           ast::DUMMY_NODE_ID),
                       span: sp(0,1)})

    }

    fn parser_done(p: Parser){
        assert_eq!(p.token.clone(), token::EOF);
    }

    #[test] fn parse_ident_pat () {
        let mut parser = string_to_parser(~"b");
        assert_eq!(parser.parse_pat(),
                   @ast::Pat{id: ast::DUMMY_NODE_ID,
                             node: ast::PatIdent(
                                ast::BindByValue(ast::MutImmutable),
                                ast::Path {
                                    span:sp(0,1),
                                    global:false,
                                    segments: ~[
                                        ast::PathSegment {
                                            identifier: str_to_ident("b"),
                                            lifetimes: opt_vec::Empty,
                                            types: opt_vec::Empty,
                                        }
                                    ],
                                },
                                None /* no idea */),
                             span: sp(0,1)});
        parser_done(parser);
    }

    // check the contents of the tt manually:
    #[test] fn parse_fundecl () {
        // this test depends on the intern order of "fn" and "int"
        assert_eq!(string_to_item(~"fn a (b : int) { b; }"),
                  Some(
                      @ast::Item{ident:str_to_ident("a"),
                            attrs:~[],
                            id: ast::DUMMY_NODE_ID,
                            node: ast::ItemFn(ast::P(ast::FnDecl {
                                inputs: ~[ast::Arg{
                                    ty: ast::P(ast::Ty{id: ast::DUMMY_NODE_ID,
                                                       node: ast::TyPath(ast::Path{
                                        span:sp(10,13),
                                        global:false,
                                        segments: ~[
                                            ast::PathSegment {
                                                identifier:
                                                    str_to_ident("int"),
                                                lifetimes: opt_vec::Empty,
                                                types: opt_vec::Empty,
                                            }
                                        ],
                                        }, None, ast::DUMMY_NODE_ID),
                                        span:sp(10,13)
                                    }),
                                    pat: @ast::Pat {
                                        id: ast::DUMMY_NODE_ID,
                                        node: ast::PatIdent(
                                            ast::BindByValue(ast::MutImmutable),
                                            ast::Path {
                                                span:sp(6,7),
                                                global:false,
                                                segments: ~[
                                                    ast::PathSegment {
                                                        identifier:
                                                            str_to_ident("b"),
                                                        lifetimes: opt_vec::Empty,
                                                        types: opt_vec::Empty,
                                                    }
                                                ],
                                            },
                                            None // no idea
                                        ),
                                        span: sp(6,7)
                                    },
                                    id: ast::DUMMY_NODE_ID
                                }],
                                output: ast::P(ast::Ty{id: ast::DUMMY_NODE_ID,
                                                       node: ast::TyNil,
                                                       span:sp(15,15)}), // not sure
                                cf: ast::Return,
                                variadic: false
                            }),
                                    ast::ImpureFn,
                                    abi::AbiSet::Rust(),
                                    ast::Generics{ // no idea on either of these:
                                        lifetimes: opt_vec::Empty,
                                        ty_params: opt_vec::Empty,
                                    },
                                    ast::P(ast::Block {
                                        view_items: ~[],
                                        stmts: ~[@Spanned{
                                            node: ast::StmtSemi(@ast::Expr{
                                                id: ast::DUMMY_NODE_ID,
                                                node: ast::ExprPath(
                                                      ast::Path{
                                                        span:sp(17,18),
                                                        global:false,
                                                        segments: ~[
                                                            ast::PathSegment {
                                                                identifier:
                                                                str_to_ident(
                                                                    "b"),
                                                                lifetimes:
                                                                opt_vec::Empty,
                                                                types:
                                                                opt_vec::Empty
                                                            }
                                                        ],
                                                      }),
                                                span: sp(17,18)},
                                                ast::DUMMY_NODE_ID),
                                            span: sp(17,18)}],
                                        expr: None,
                                        id: ast::DUMMY_NODE_ID,
                                        rules: ast::DefaultBlock, // no idea
                                        span: sp(15,21),
                                    })),
                            vis: ast::Inherited,
                            span: sp(0,21)}));
    }


    #[test] fn parse_exprs () {
        // just make sure that they parse....
        string_to_expr(~"3 + 4");
        string_to_expr(~"a::z.froob(b,@(987+3))");
    }

    #[test] fn attrs_fix_bug () {
        string_to_item(~"pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
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
