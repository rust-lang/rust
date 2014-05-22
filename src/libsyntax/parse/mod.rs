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
use diagnostic::{SpanHandler, mk_span_handler, default_handler, Auto};
use parse::attr::ParserAttr;
use parse::parser::Parser;

use std::cell::RefCell;
use std::io::File;
use std::rc::Rc;
use std::str;

pub mod lexer;
pub mod parser;
pub mod token;
pub mod comments;
pub mod attr;

pub mod common;
pub mod classify;
pub mod obsolete;

// info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: SpanHandler, // better be the same as the one in the reader!
    /// Used to determine and report recursive mod inclusions
    included_mod_stack: RefCell<Vec<Path>>,
}

pub fn new_parse_sess() -> ParseSess {
    ParseSess {
        span_diagnostic: mk_span_handler(default_handler(Auto), CodeMap::new()),
        included_mod_stack: RefCell::new(Vec::new()),
    }
}

pub fn new_parse_sess_special_handler(sh: SpanHandler) -> ParseSess {
    ParseSess {
        span_diagnostic: sh,
        included_mod_stack: RefCell::new(Vec::new()),
    }
}

// a bunch of utility functions of the form parse_<thing>_from_<source>
// where <thing> includes crate, expr, item, stmt, tts, and one that
// uses a HOF to parse anything, and <source> includes file and
// source_str.

pub fn parse_crate_from_file(
    input: &Path,
    cfg: ast::CrateConfig,
    sess: &ParseSess
) -> ast::Crate {
    new_parser_from_file(sess, cfg, input).parse_crate_mod()
    // why is there no p.abort_if_errors here?
}

pub fn parse_crate_attrs_from_file(
    input: &Path,
    cfg: ast::CrateConfig,
    sess: &ParseSess
) -> Vec<ast::Attribute> {
    let mut parser = new_parser_from_file(sess, cfg, input);
    let (inner, _) = parser.parse_inner_attrs_and_next();
    inner
}

pub fn parse_crate_from_source_str(name: StrBuf,
                                   source: StrBuf,
                                   cfg: ast::CrateConfig,
                                   sess: &ParseSess)
                                   -> ast::Crate {
    let mut p = new_parser_from_source_str(sess,
                                           cfg,
                                           name,
                                           source);
    maybe_aborted(p.parse_crate_mod(),p)
}

pub fn parse_crate_attrs_from_source_str(name: StrBuf,
                                         source: StrBuf,
                                         cfg: ast::CrateConfig,
                                         sess: &ParseSess)
                                         -> Vec<ast::Attribute> {
    let mut p = new_parser_from_source_str(sess,
                                           cfg,
                                           name,
                                           source);
    let (inner, _) = maybe_aborted(p.parse_inner_attrs_and_next(),p);
    inner
}

pub fn parse_expr_from_source_str(name: StrBuf,
                                  source: StrBuf,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> @ast::Expr {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_expr(), p)
}

pub fn parse_item_from_source_str(name: StrBuf,
                                  source: StrBuf,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> Option<@ast::Item> {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    let attrs = p.parse_outer_attributes();
    maybe_aborted(p.parse_item(attrs),p)
}

pub fn parse_meta_from_source_str(name: StrBuf,
                                  source: StrBuf,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> @ast::MetaItem {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_meta_item(),p)
}

pub fn parse_stmt_from_source_str(name: StrBuf,
                                  source: StrBuf,
                                  cfg: ast::CrateConfig,
                                  attrs: Vec<ast::Attribute> ,
                                  sess: &ParseSess)
                                  -> @ast::Stmt {
    let mut p = new_parser_from_source_str(
        sess,
        cfg,
        name,
        source
    );
    maybe_aborted(p.parse_stmt(attrs),p)
}

pub fn parse_tts_from_source_str(name: StrBuf,
                                 source: StrBuf,
                                 cfg: ast::CrateConfig,
                                 sess: &ParseSess)
                                 -> Vec<ast::TokenTree> {
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
pub fn new_parser_from_source_str<'a>(sess: &'a ParseSess,
                                      cfg: ast::CrateConfig,
                                      name: StrBuf,
                                      source: StrBuf)
                                      -> Parser<'a> {
    filemap_to_parser(sess, string_to_filemap(sess, source, name), cfg)
}

/// Create a new parser, handling errors as appropriate
/// if the file doesn't exist
pub fn new_parser_from_file<'a>(sess: &'a ParseSess,
                                cfg: ast::CrateConfig,
                                path: &Path) -> Parser<'a> {
    filemap_to_parser(sess, file_to_filemap(sess, path, None), cfg)
}

/// Given a session, a crate config, a path, and a span, add
/// the file at the given path to the codemap, and return a parser.
/// On an error, use the given span as the source of the problem.
pub fn new_sub_parser_from_file<'a>(sess: &'a ParseSess,
                                    cfg: ast::CrateConfig,
                                    path: &Path,
                                    owns_directory: bool,
                                    module_name: Option<StrBuf>,
                                    sp: Span) -> Parser<'a> {
    let mut p = filemap_to_parser(sess, file_to_filemap(sess, path, Some(sp)), cfg);
    p.owns_directory = owns_directory;
    p.root_module_name = module_name;
    p
}

/// Given a filemap and config, return a parser
pub fn filemap_to_parser<'a>(sess: &'a ParseSess,
                             filemap: Rc<FileMap>,
                             cfg: ast::CrateConfig) -> Parser<'a> {
    tts_to_parser(sess, filemap_to_tts(sess, filemap), cfg)
}

// must preserve old name for now, because quote! from the *existing*
// compiler expands into it
pub fn new_parser_from_tts<'a>(sess: &'a ParseSess,
                               cfg: ast::CrateConfig,
                               tts: Vec<ast::TokenTree>) -> Parser<'a> {
    tts_to_parser(sess, tts, cfg)
}


// base abstractions

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's codemap and return the new filemap.
pub fn file_to_filemap(sess: &ParseSess, path: &Path, spanopt: Option<Span>)
    -> Rc<FileMap> {
    let err = |msg: &str| {
        match spanopt {
            Some(sp) => sess.span_diagnostic.span_fatal(sp, msg),
            None => sess.span_diagnostic.handler().fatal(msg),
        }
    };
    let bytes = match File::open(path).read_to_end() {
        Ok(bytes) => bytes,
        Err(e) => {
            err(format!("couldn't read {}: {}",
                        path.display(),
                        e).as_slice());
            unreachable!()
        }
    };
    match str::from_utf8(bytes.as_slice()) {
        Some(s) => {
            return string_to_filemap(sess, s.to_strbuf(),
                                     path.as_str().unwrap().to_strbuf())
        }
        None => {
            err(format!("{} is not UTF-8 encoded", path.display()).as_slice())
        }
    }
    unreachable!()
}

// given a session and a string, add the string to
// the session's codemap and return the new filemap
pub fn string_to_filemap(sess: &ParseSess, source: StrBuf, path: StrBuf)
                         -> Rc<FileMap> {
    sess.span_diagnostic.cm.new_filemap(path, source)
}

// given a filemap, produce a sequence of token-trees
pub fn filemap_to_tts(sess: &ParseSess, filemap: Rc<FileMap>)
    -> Vec<ast::TokenTree> {
    // it appears to me that the cfg doesn't matter here... indeed,
    // parsing tt's probably shouldn't require a parser at all.
    let cfg = Vec::new();
    let srdr = lexer::new_string_reader(&sess.span_diagnostic, filemap);
    let mut p1 = Parser(sess, cfg, box srdr);
    p1.parse_all_token_trees()
}

// given tts and cfg, produce a parser
pub fn tts_to_parser<'a>(sess: &'a ParseSess,
                         tts: Vec<ast::TokenTree>,
                         cfg: ast::CrateConfig) -> Parser<'a> {
    let trdr = lexer::new_tt_reader(&sess.span_diagnostic, None, tts);
    Parser(sess, cfg, box trdr)
}

// abort if necessary
pub fn maybe_aborted<T>(result: T, mut p: Parser) -> T {
    p.abort_if_errors();
    result
}



#[cfg(test)]
mod test {
    use super::*;
    use serialize::{json, Encodable};
    use std::io;
    use std::io::MemWriter;
    use std::str;
    use codemap::{Span, BytePos, Spanned};
    use owned_slice::OwnedSlice;
    use ast;
    use abi;
    use parse::parser::Parser;
    use parse::token::{str_to_ident};
    use util::parser_testing::{string_to_tts, string_to_parser};
    use util::parser_testing::{string_to_expr, string_to_item};
    use util::parser_testing::string_to_stmt;

    fn to_json_str<'a, E: Encodable<json::Encoder<'a>, io::IoError>>(val: &E) -> StrBuf {
        let mut writer = MemWriter::new();
        let mut encoder = json::Encoder::new(&mut writer as &mut io::Writer);
        let _ = val.encode(&mut encoder);
        str::from_utf8(writer.unwrap().as_slice()).unwrap().to_strbuf()
    }

    // produce a codemap::span
    fn sp(a: u32, b: u32) -> Span {
        Span{lo:BytePos(a),hi:BytePos(b),expn_info:None}
    }

    #[test] fn path_exprs_1() {
        assert!(string_to_expr("a".to_strbuf()) ==
                   @ast::Expr{
                    id: ast::DUMMY_NODE_ID,
                    node: ast::ExprPath(ast::Path {
                        span: sp(0, 1),
                        global: false,
                        segments: vec!(
                            ast::PathSegment {
                                identifier: str_to_ident("a"),
                                lifetimes: Vec::new(),
                                types: OwnedSlice::empty(),
                            }
                        ),
                    }),
                    span: sp(0, 1)
                   })
    }

    #[test] fn path_exprs_2 () {
        assert!(string_to_expr("::a::b".to_strbuf()) ==
                   @ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    node: ast::ExprPath(ast::Path {
                            span: sp(0, 6),
                            global: true,
                            segments: vec!(
                                ast::PathSegment {
                                    identifier: str_to_ident("a"),
                                    lifetimes: Vec::new(),
                                    types: OwnedSlice::empty(),
                                },
                                ast::PathSegment {
                                    identifier: str_to_ident("b"),
                                    lifetimes: Vec::new(),
                                    types: OwnedSlice::empty(),
                                }
                            )
                        }),
                    span: sp(0, 6)
                   })
    }

    #[should_fail]
    #[test] fn bad_path_expr_1() {
        string_to_expr("::abc::def::return".to_strbuf());
    }

    // check the token-tree-ization of macros
    #[test] fn string_to_tts_macro () {
        let tts = string_to_tts("macro_rules! zip (($a)=>($a))".to_strbuf());
        let tts: &[ast::TokenTree] = tts.as_slice();
        match tts {
            [ast::TTTok(_,_),
             ast::TTTok(_,token::NOT),
             ast::TTTok(_,_),
             ast::TTDelim(ref delim_elts)] => {
                let delim_elts: &[ast::TokenTree] = delim_elts.as_slice();
                match delim_elts {
                    [ast::TTTok(_,token::LPAREN),
                     ast::TTDelim(ref first_set),
                     ast::TTTok(_,token::FAT_ARROW),
                     ast::TTDelim(ref second_set),
                     ast::TTTok(_,token::RPAREN)] => {
                        let first_set: &[ast::TokenTree] =
                            first_set.as_slice();
                        match first_set {
                            [ast::TTTok(_,token::LPAREN),
                             ast::TTTok(_,token::DOLLAR),
                             ast::TTTok(_,_),
                             ast::TTTok(_,token::RPAREN)] => {
                                let second_set: &[ast::TokenTree] =
                                    second_set.as_slice();
                                match second_set {
                                    [ast::TTTok(_,token::LPAREN),
                                     ast::TTTok(_,token::DOLLAR),
                                     ast::TTTok(_,_),
                                     ast::TTTok(_,token::RPAREN)] => {
                                        assert_eq!("correct","correct")
                                    }
                                    _ => assert_eq!("wrong 4","correct")
                                }
                            },
                            _ => {
                                error!("failing value 3: {:?}",first_set);
                                assert_eq!("wrong 3","correct")
                            }
                        }
                    },
                    _ => {
                        error!("failing value 2: {:?}",delim_elts);
                        assert_eq!("wrong","correct");
                    }
                }
            },
            _ => {
                error!("failing value: {:?}",tts);
                assert_eq!("wrong 1","correct");
            }
        }
    }

    #[test] fn string_to_tts_1 () {
        let tts = string_to_tts("fn a (b : int) { b; }".to_strbuf());
        assert_eq!(to_json_str(&tts),
        "[\
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
]".to_strbuf()
        );
    }

    #[test] fn ret_expr() {
        assert!(string_to_expr("return d".to_strbuf()) ==
                   @ast::Expr{
                    id: ast::DUMMY_NODE_ID,
                    node:ast::ExprRet(Some(@ast::Expr{
                        id: ast::DUMMY_NODE_ID,
                        node:ast::ExprPath(ast::Path{
                            span: sp(7, 8),
                            global: false,
                            segments: vec!(
                                ast::PathSegment {
                                    identifier: str_to_ident("d"),
                                    lifetimes: Vec::new(),
                                    types: OwnedSlice::empty(),
                                }
                            ),
                        }),
                        span:sp(7,8)
                    })),
                    span:sp(0,8)
                   })
    }

    #[test] fn parse_stmt_1 () {
        assert!(string_to_stmt("b;".to_strbuf()) ==
                   @Spanned{
                       node: ast::StmtExpr(@ast::Expr {
                           id: ast::DUMMY_NODE_ID,
                           node: ast::ExprPath(ast::Path {
                               span:sp(0,1),
                               global:false,
                               segments: vec!(
                                ast::PathSegment {
                                    identifier: str_to_ident("b"),
                                    lifetimes: Vec::new(),
                                    types: OwnedSlice::empty(),
                                }
                               ),
                            }),
                           span: sp(0,1)},
                                           ast::DUMMY_NODE_ID),
                       span: sp(0,1)})

    }

    fn parser_done(p: Parser){
        assert_eq!(p.token.clone(), token::EOF);
    }

    #[test] fn parse_ident_pat () {
        let sess = new_parse_sess();
        let mut parser = string_to_parser(&sess, "b".to_strbuf());
        assert!(parser.parse_pat() ==
                   @ast::Pat{id: ast::DUMMY_NODE_ID,
                             node: ast::PatIdent(
                                ast::BindByValue(ast::MutImmutable),
                                ast::Path {
                                    span:sp(0,1),
                                    global:false,
                                    segments: vec!(
                                        ast::PathSegment {
                                            identifier: str_to_ident("b"),
                                            lifetimes: Vec::new(),
                                            types: OwnedSlice::empty(),
                                        }
                                    ),
                                },
                                None /* no idea */),
                             span: sp(0,1)});
        parser_done(parser);
    }

    // check the contents of the tt manually:
    #[test] fn parse_fundecl () {
        // this test depends on the intern order of "fn" and "int"
        assert!(string_to_item("fn a (b : int) { b; }".to_strbuf()) ==
                  Some(
                      @ast::Item{ident:str_to_ident("a"),
                            attrs:Vec::new(),
                            id: ast::DUMMY_NODE_ID,
                            node: ast::ItemFn(ast::P(ast::FnDecl {
                                inputs: vec!(ast::Arg{
                                    ty: ast::P(ast::Ty{id: ast::DUMMY_NODE_ID,
                                                       node: ast::TyPath(ast::Path{
                                        span:sp(10,13),
                                        global:false,
                                        segments: vec!(
                                            ast::PathSegment {
                                                identifier:
                                                    str_to_ident("int"),
                                                lifetimes: Vec::new(),
                                                types: OwnedSlice::empty(),
                                            }
                                        ),
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
                                                segments: vec!(
                                                    ast::PathSegment {
                                                        identifier:
                                                            str_to_ident("b"),
                                                        lifetimes: Vec::new(),
                                                        types: OwnedSlice::empty(),
                                                    }
                                                ),
                                            },
                                            None // no idea
                                        ),
                                        span: sp(6,7)
                                    },
                                    id: ast::DUMMY_NODE_ID
                                }),
                                output: ast::P(ast::Ty{id: ast::DUMMY_NODE_ID,
                                                       node: ast::TyNil,
                                                       span:sp(15,15)}), // not sure
                                cf: ast::Return,
                                variadic: false
                            }),
                                    ast::NormalFn,
                                    abi::Rust,
                                    ast::Generics{ // no idea on either of these:
                                        lifetimes: Vec::new(),
                                        ty_params: OwnedSlice::empty(),
                                    },
                                    ast::P(ast::Block {
                                        view_items: Vec::new(),
                                        stmts: vec!(@Spanned{
                                            node: ast::StmtSemi(@ast::Expr{
                                                id: ast::DUMMY_NODE_ID,
                                                node: ast::ExprPath(
                                                      ast::Path{
                                                        span:sp(17,18),
                                                        global:false,
                                                        segments: vec!(
                                                            ast::PathSegment {
                                                                identifier:
                                                                str_to_ident(
                                                                    "b"),
                                                                lifetimes:
                                                                Vec::new(),
                                                                types:
                                                                OwnedSlice::empty()
                                                            }
                                                        ),
                                                      }),
                                                span: sp(17,18)},
                                                ast::DUMMY_NODE_ID),
                                            span: sp(17,19)}),
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
        string_to_expr("3 + 4".to_strbuf());
        string_to_expr("a::z.froob(b,@(987+3))".to_strbuf());
    }

    #[test] fn attrs_fix_bug () {
        string_to_item("pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<@Writer, StrBuf> {
    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
}".to_strbuf());
    }

}
