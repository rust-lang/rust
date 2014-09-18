// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use ptr::P;

use std::cell::{Cell, RefCell};
use std::io::File;
use std::rc::Rc;
use std::str;
use std::iter;

pub mod lexer;
pub mod parser;
pub mod token;
pub mod attr;

pub mod common;
pub mod classify;
pub mod obsolete;

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: SpanHandler, // better be the same as the one in the reader!
    /// Used to determine and report recursive mod inclusions
    included_mod_stack: RefCell<Vec<Path>>,
    pub node_id: Cell<ast::NodeId>,
}

pub fn new_parse_sess() -> ParseSess {
    ParseSess {
        span_diagnostic: mk_span_handler(default_handler(Auto, None), CodeMap::new()),
        included_mod_stack: RefCell::new(Vec::new()),
        node_id: Cell::new(1),
    }
}

pub fn new_parse_sess_special_handler(sh: SpanHandler) -> ParseSess {
    ParseSess {
        span_diagnostic: sh,
        included_mod_stack: RefCell::new(Vec::new()),
        node_id: Cell::new(1),
    }
}

impl ParseSess {
    pub fn next_node_id(&self) -> ast::NodeId {
        self.reserve_node_ids(1)
    }
    pub fn reserve_node_ids(&self, count: ast::NodeId) -> ast::NodeId {
        let v = self.node_id.get();

        match v.checked_add(&count) {
            Some(next) => { self.node_id.set(next); }
            None => fail!("Input too large, ran out of node ids!")
        }

        v
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

pub fn parse_crate_from_source_str(name: String,
                                   source: String,
                                   cfg: ast::CrateConfig,
                                   sess: &ParseSess)
                                   -> ast::Crate {
    let mut p = new_parser_from_source_str(sess,
                                           cfg,
                                           name,
                                           source);
    maybe_aborted(p.parse_crate_mod(),p)
}

pub fn parse_crate_attrs_from_source_str(name: String,
                                         source: String,
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

pub fn parse_expr_from_source_str(name: String,
                                  source: String,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> P<ast::Expr> {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_expr(), p)
}

pub fn parse_item_from_source_str(name: String,
                                  source: String,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> Option<P<ast::Item>> {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_item_with_outer_attributes(),p)
}

pub fn parse_meta_from_source_str(name: String,
                                  source: String,
                                  cfg: ast::CrateConfig,
                                  sess: &ParseSess)
                                  -> P<ast::MetaItem> {
    let mut p = new_parser_from_source_str(sess, cfg, name, source);
    maybe_aborted(p.parse_meta_item(),p)
}

pub fn parse_stmt_from_source_str(name: String,
                                  source: String,
                                  cfg: ast::CrateConfig,
                                  attrs: Vec<ast::Attribute> ,
                                  sess: &ParseSess)
                                  -> P<ast::Stmt> {
    let mut p = new_parser_from_source_str(
        sess,
        cfg,
        name,
        source
    );
    maybe_aborted(p.parse_stmt(attrs),p)
}

// Note: keep in sync with `with_hygiene::parse_tts_from_source_str`
// until #16472 is resolved.
pub fn parse_tts_from_source_str(name: String,
                                 source: String,
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

// Note: keep in sync with `with_hygiene::new_parser_from_source_str`
// until #16472 is resolved.
// Create a new parser from a source string
pub fn new_parser_from_source_str<'a>(sess: &'a ParseSess,
                                      cfg: ast::CrateConfig,
                                      name: String,
                                      source: String)
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
                                    module_name: Option<String>,
                                    sp: Span) -> Parser<'a> {
    let mut p = filemap_to_parser(sess, file_to_filemap(sess, path, Some(sp)), cfg);
    p.owns_directory = owns_directory;
    p.root_module_name = module_name;
    p
}

// Note: keep this in sync with `with_hygiene::filemap_to_parser` until
// #16472 is resolved.
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
            return string_to_filemap(sess, s.to_string(),
                                     path.as_str().unwrap().to_string())
        }
        None => {
            err(format!("{} is not UTF-8 encoded", path.display()).as_slice())
        }
    }
    unreachable!()
}

/// Given a session and a string, add the string to
/// the session's codemap and return the new filemap
pub fn string_to_filemap(sess: &ParseSess, source: String, path: String)
                         -> Rc<FileMap> {
    sess.span_diagnostic.cm.new_filemap(path, source)
}

// Note: keep this in sync with `with_hygiene::filemap_to_tts` (apart
// from the StringReader constructor), until #16472 is resolved.
/// Given a filemap, produce a sequence of token-trees
pub fn filemap_to_tts(sess: &ParseSess, filemap: Rc<FileMap>)
    -> Vec<ast::TokenTree> {
    // it appears to me that the cfg doesn't matter here... indeed,
    // parsing tt's probably shouldn't require a parser at all.
    let cfg = Vec::new();
    let srdr = lexer::StringReader::new(&sess.span_diagnostic, filemap);
    let mut p1 = Parser::new(sess, cfg, box srdr);
    p1.parse_all_token_trees()
}

/// Given tts and cfg, produce a parser
pub fn tts_to_parser<'a>(sess: &'a ParseSess,
                         tts: Vec<ast::TokenTree>,
                         cfg: ast::CrateConfig) -> Parser<'a> {
    let trdr = lexer::new_tt_reader(&sess.span_diagnostic, None, tts);
    Parser::new(sess, cfg, box trdr)
}

// FIXME (Issue #16472): The `with_hygiene` mod should go away after
// ToToken impls are revised to go directly to token-trees.
pub mod with_hygiene {
    use ast;
    use codemap::FileMap;
    use parse::parser::Parser;
    use std::rc::Rc;
    use super::ParseSess;
    use super::{maybe_aborted, string_to_filemap, tts_to_parser};

    // Note: keep this in sync with `super::parse_tts_from_source_str` until
    // #16472 is resolved.
    pub fn parse_tts_from_source_str(name: String,
                                     source: String,
                                     cfg: ast::CrateConfig,
                                     sess: &ParseSess) -> Vec<ast::TokenTree> {
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

    // Note: keep this in sync with `super::new_parser_from_source_str` until
    // #16472 is resolved.
    // Create a new parser from a source string
    fn new_parser_from_source_str<'a>(sess: &'a ParseSess,
                                      cfg: ast::CrateConfig,
                                      name: String,
                                      source: String) -> Parser<'a> {
        filemap_to_parser(sess, string_to_filemap(sess, source, name), cfg)
    }

    // Note: keep this in sync with `super::filemap_to_parserr` until
    // #16472 is resolved.
    /// Given a filemap and config, return a parser
    fn filemap_to_parser<'a>(sess: &'a ParseSess,
                             filemap: Rc<FileMap>,
                             cfg: ast::CrateConfig) -> Parser<'a> {
        tts_to_parser(sess, filemap_to_tts(sess, filemap), cfg)
    }

    // Note: keep this in sync with `super::filemap_to_tts` until
    // #16472 is resolved.
    /// Given a filemap, produce a sequence of token-trees
    fn filemap_to_tts(sess: &ParseSess, filemap: Rc<FileMap>)
                      -> Vec<ast::TokenTree> {
        // it appears to me that the cfg doesn't matter here... indeed,
        // parsing tt's probably shouldn't require a parser at all.
        use super::lexer::make_reader_with_embedded_idents as make_reader;
        let cfg = Vec::new();
        let srdr = make_reader(&sess.span_diagnostic, filemap);
        let mut p1 = Parser::new(sess, cfg, box srdr);
        p1.parse_all_token_trees()
    }
}

/// Abort if necessary
pub fn maybe_aborted<T>(result: T, mut p: Parser) -> T {
    p.abort_if_errors();
    result
}

/// Parse a string representing a character literal into its final form.
/// Rather than just accepting/rejecting a given literal, unescapes it as
/// well. Can take any slice prefixed by a character escape. Returns the
/// character and the number of characters consumed.
pub fn char_lit(lit: &str) -> (char, int) {
    use std::{num, char};

    let mut chars = lit.chars();
    let c = match (chars.next(), chars.next()) {
        (Some(c), None) if c != '\\' => return (c, 1),
        (Some('\\'), Some(c)) => match c {
            '"' => Some('"'),
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            '\\' => Some('\\'),
            '\'' => Some('\''),
            '0' => Some('\0'),
            _ => { None }
        },
        _ => fail!("lexer accepted invalid char escape `{}`", lit)
    };

    match c {
        Some(x) => return (x, 2),
        None => { }
    }

    let msg = format!("lexer should have rejected a bad character escape {}", lit);
    let msg2 = msg.as_slice();

    let esc: |uint| -> Option<(char, int)> = |len|
        num::from_str_radix(lit.slice(2, len), 16)
        .and_then(char::from_u32)
        .map(|x| (x, len as int));

    // Unicode escapes
    return match lit.as_bytes()[1] as char {
        'x' | 'X' => esc(4),
        'u' => esc(6),
        'U' => esc(10),
        _ => None,
    }.expect(msg2);
}

/// Parse a string representing a string literal into its final form. Does
/// unescaping.
pub fn str_lit(lit: &str) -> String {
    debug!("parse_str_lit: given {}", lit.escape_default());
    let mut res = String::with_capacity(lit.len());

    // FIXME #8372: This could be a for-loop if it didn't borrow the iterator
    let error = |i| format!("lexer should have rejected {} at {}", lit, i);

    /// Eat everything up to a non-whitespace
    fn eat<'a>(it: &mut iter::Peekable<(uint, char), str::CharOffsets<'a>>) {
        loop {
            match it.peek().map(|x| x.val1()) {
                Some(' ') | Some('\n') | Some('\r') | Some('\t') => {
                    it.next();
                },
                _ => { break; }
            }
        }
    }

    let mut chars = lit.char_indices().peekable();
    loop {
        match chars.next() {
            Some((i, c)) => {
                match c {
                    '\\' => {
                        let ch = chars.peek().unwrap_or_else(|| {
                            fail!("{}", error(i).as_slice())
                        }).val1();

                        if ch == '\n' {
                            eat(&mut chars);
                        } else if ch == '\r' {
                            chars.next();
                            let ch = chars.peek().unwrap_or_else(|| {
                                fail!("{}", error(i).as_slice())
                            }).val1();

                            if ch != '\n' {
                                fail!("lexer accepted bare CR");
                            }
                            eat(&mut chars);
                        } else {
                            // otherwise, a normal escape
                            let (c, n) = char_lit(lit.slice_from(i));
                            for _ in range(0, n - 1) { // we don't need to move past the first \
                                chars.next();
                            }
                            res.push_char(c);
                        }
                    },
                    '\r' => {
                        let ch = chars.peek().unwrap_or_else(|| {
                            fail!("{}", error(i).as_slice())
                        }).val1();

                        if ch != '\n' {
                            fail!("lexer accepted bare CR");
                        }
                        chars.next();
                        res.push_char('\n');
                    }
                    c => res.push_char(c),
                }
            },
            None => break
        }
    }

    res.shrink_to_fit(); // probably not going to do anything, unless there was an escape.
    debug!("parse_str_lit: returning {}", res);
    res
}

/// Parse a string representing a raw string literal into its final form. The
/// only operation this does is convert embedded CRLF into a single LF.
pub fn raw_str_lit(lit: &str) -> String {
    debug!("raw_str_lit: given {}", lit.escape_default());
    let mut res = String::with_capacity(lit.len());

    // FIXME #8372: This could be a for-loop if it didn't borrow the iterator
    let mut chars = lit.chars().peekable();
    loop {
        match chars.next() {
            Some(c) => {
                if c == '\r' {
                    if *chars.peek().unwrap() != '\n' {
                        fail!("lexer accepted bare CR");
                    }
                    chars.next();
                    res.push_char('\n');
                } else {
                    res.push_char(c);
                }
            },
            None => break
        }
    }

    res.shrink_to_fit();
    res
}

pub fn float_lit(s: &str) -> ast::Lit_ {
    debug!("float_lit: {}", s);
    // FIXME #2252: bounds checking float literals is defered until trans
    let s2 = s.chars().filter(|&c| c != '_').collect::<String>();
    let s = s2.as_slice();

    let mut ty = None;

    if s.ends_with("f32") {
        ty = Some(ast::TyF32);
    } else if s.ends_with("f64") {
        ty = Some(ast::TyF64);
    }


    match ty {
        Some(t) => {
            ast::LitFloat(token::intern_and_get_ident(s.slice_to(s.len() - t.suffix_len())), t)
        },
        None => ast::LitFloatUnsuffixed(token::intern_and_get_ident(s))
    }
}

/// Parse a string representing a byte literal into its final form. Similar to `char_lit`
pub fn byte_lit(lit: &str) -> (u8, uint) {
    let err = |i| format!("lexer accepted invalid byte literal {} step {}", lit, i);

    if lit.len() == 1 {
        (lit.as_bytes()[0], 1)
    } else {
        assert!(lit.as_bytes()[0] == b'\\', err(0i));
        let b = match lit.as_bytes()[1] {
            b'"' => b'"',
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'\\' => b'\\',
            b'\'' => b'\'',
            b'0' => b'\0',
            _ => {
                match ::std::num::from_str_radix::<u64>(lit.slice(2, 4), 16) {
                    Some(c) =>
                        if c > 0xFF {
                            fail!(err(2))
                        } else {
                            return (c as u8, 4)
                        },
                    None => fail!(err(3))
                }
            }
        };
        return (b, 2);
    }
}

pub fn binary_lit(lit: &str) -> Rc<Vec<u8>> {
    let mut res = Vec::with_capacity(lit.len());

    // FIXME #8372: This could be a for-loop if it didn't borrow the iterator
    let error = |i| format!("lexer should have rejected {} at {}", lit, i);

    /// Eat everything up to a non-whitespace
    fn eat<'a, I: Iterator<(uint, u8)>>(it: &mut iter::Peekable<(uint, u8), I>) {
        loop {
            match it.peek().map(|x| x.val1()) {
                Some(b' ') | Some(b'\n') | Some(b'\r') | Some(b'\t') => {
                    it.next();
                },
                _ => { break; }
            }
        }
    }

    // binary literals *must* be ASCII, but the escapes don't have to be
    let mut chars = lit.bytes().enumerate().peekable();
    loop {
        match chars.next() {
            Some((i, b'\\')) => {
                let em = error(i);
                match chars.peek().expect(em.as_slice()).val1() {
                    b'\n' => eat(&mut chars),
                    b'\r' => {
                        chars.next();
                        if chars.peek().expect(em.as_slice()).val1() != b'\n' {
                            fail!("lexer accepted bare CR");
                        }
                        eat(&mut chars);
                    }
                    _ => {
                        // otherwise, a normal escape
                        let (c, n) = byte_lit(lit.slice_from(i));
                        // we don't need to move past the first \
                        for _ in range(0, n - 1) {
                            chars.next();
                        }
                        res.push(c);
                    }
                }
            },
            Some((i, b'\r')) => {
                let em = error(i);
                if chars.peek().expect(em.as_slice()).val1() != b'\n' {
                    fail!("lexer accepted bare CR");
                }
                chars.next();
                res.push(b'\n');
            }
            Some((_, c)) => res.push(c),
            None => break,
        }
    }

    Rc::new(res)
}

pub fn integer_lit(s: &str, sd: &SpanHandler, sp: Span) -> ast::Lit_ {
    // s can only be ascii, byte indexing is fine

    let s2 = s.chars().filter(|&c| c != '_').collect::<String>();
    let mut s = s2.as_slice();

    debug!("parse_integer_lit: {}", s);

    if s.len() == 1 {
        let n = (s.char_at(0)).to_digit(10).unwrap();
        return ast::LitInt(n as u64, ast::UnsuffixedIntLit(ast::Sign::new(n)));
    }

    let mut base = 10;
    let orig = s;
    let mut ty = ast::UnsuffixedIntLit(ast::Plus);

    if s.char_at(0) == '0' {
        match s.char_at(1) {
            'x' => base = 16,
            'o' => base = 8,
            'b' => base = 2,
            _ => { }
        }
    }

    if base != 10 {
        s = s.slice_from(2);
    }

    let last = s.len() - 1;
    match s.char_at(last) {
        'i' => ty = ast::SignedIntLit(ast::TyI, ast::Plus),
        'u' => ty = ast::UnsignedIntLit(ast::TyU),
        '8' => {
            if s.len() > 2 {
                match s.char_at(last - 1) {
                    'i' => ty = ast::SignedIntLit(ast::TyI8, ast::Plus),
                    'u' => ty = ast::UnsignedIntLit(ast::TyU8),
                    _ => { }
                }
            }
        },
        '6' => {
            if s.len() > 3 && s.char_at(last - 1) == '1' {
                match s.char_at(last - 2) {
                    'i' => ty = ast::SignedIntLit(ast::TyI16, ast::Plus),
                    'u' => ty = ast::UnsignedIntLit(ast::TyU16),
                    _ => { }
                }
            }
        },
        '2' => {
            if s.len() > 3 && s.char_at(last - 1) == '3' {
                match s.char_at(last - 2) {
                    'i' => ty = ast::SignedIntLit(ast::TyI32, ast::Plus),
                    'u' => ty = ast::UnsignedIntLit(ast::TyU32),
                    _ => { }
                }
            }
        },
        '4' => {
            if s.len() > 3 && s.char_at(last - 1) == '6' {
                match s.char_at(last - 2) {
                    'i' => ty = ast::SignedIntLit(ast::TyI64, ast::Plus),
                    'u' => ty = ast::UnsignedIntLit(ast::TyU64),
                    _ => { }
                }
            }
        },
        _ => { }
    }

    debug!("The suffix is {}, base {}, the new string is {}, the original \
           string was {}", ty, base, s, orig);

    s = s.slice_to(s.len() - ty.suffix_len());

    let res: u64 = match ::std::num::from_str_radix(s, base) {
        Some(r) => r,
        None => { sd.span_err(sp, "int literal is too large"); 0 }
    };

    // adjust the sign
    let sign = ast::Sign::new(res);
    match ty {
        ast::SignedIntLit(t, _) => ast::LitInt(res, ast::SignedIntLit(t, sign)),
        ast::UnsuffixedIntLit(_) => ast::LitInt(res, ast::UnsuffixedIntLit(sign)),
        us@ast::UnsignedIntLit(_) => ast::LitInt(res, us)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serialize::json;
    use codemap::{Span, BytePos, Spanned, NO_EXPANSION};
    use owned_slice::OwnedSlice;
    use ast;
    use abi;
    use attr;
    use attr::AttrMetaMethods;
    use parse::parser::Parser;
    use parse::token::{str_to_ident};
    use ptr::P;
    use util::parser_testing::{string_to_tts, string_to_parser};
    use util::parser_testing::{string_to_expr, string_to_item};
    use util::parser_testing::string_to_stmt;

    // produce a codemap::span
    fn sp(a: u32, b: u32) -> Span {
        Span {lo: BytePos(a), hi: BytePos(b), expn_id: NO_EXPANSION}
    }

    #[test] fn path_exprs_1() {
        assert!(string_to_expr("a".to_string()) ==
                   P(ast::Expr{
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
                   }))
    }

    #[test] fn path_exprs_2 () {
        assert!(string_to_expr("::a::b".to_string()) ==
                   P(ast::Expr {
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
                   }))
    }

    #[should_fail]
    #[test] fn bad_path_expr_1() {
        string_to_expr("::abc::def::return".to_string());
    }

    // check the token-tree-ization of macros
    #[test] fn string_to_tts_macro () {
        let tts = string_to_tts("macro_rules! zip (($a)=>($a))".to_string());
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
        let tts = string_to_tts("fn a (b : int) { b; }".to_string());
        assert_eq!(json::encode(&tts),
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
]".to_string()
        );
    }

    #[test] fn ret_expr() {
        assert!(string_to_expr("return d".to_string()) ==
                   P(ast::Expr{
                    id: ast::DUMMY_NODE_ID,
                    node:ast::ExprRet(Some(P(ast::Expr{
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
                    }))),
                    span:sp(0,8)
                   }))
    }

    #[test] fn parse_stmt_1 () {
        assert!(string_to_stmt("b;".to_string()) ==
                   P(Spanned{
                       node: ast::StmtExpr(P(ast::Expr {
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
                           span: sp(0,1)}),
                                           ast::DUMMY_NODE_ID),
                       span: sp(0,1)}))

    }

    fn parser_done(p: Parser){
        assert_eq!(p.token.clone(), token::EOF);
    }

    #[test] fn parse_ident_pat () {
        let sess = new_parse_sess();
        let mut parser = string_to_parser(&sess, "b".to_string());
        assert!(parser.parse_pat()
                == P(ast::Pat{
                id: ast::DUMMY_NODE_ID,
                node: ast::PatIdent(ast::BindByValue(ast::MutImmutable),
                                    Spanned{ span:sp(0, 1),
                                             node: str_to_ident("b")
                    },
                                    None),
                span: sp(0,1)}));
        parser_done(parser);
    }

    // check the contents of the tt manually:
    #[test] fn parse_fundecl () {
        // this test depends on the intern order of "fn" and "int"
        assert!(string_to_item("fn a (b : int) { b; }".to_string()) ==
                  Some(
                      P(ast::Item{ident:str_to_ident("a"),
                            attrs:Vec::new(),
                            id: ast::DUMMY_NODE_ID,
                            node: ast::ItemFn(P(ast::FnDecl {
                                inputs: vec!(ast::Arg{
                                    ty: P(ast::Ty{id: ast::DUMMY_NODE_ID,
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
                                    pat: P(ast::Pat {
                                        id: ast::DUMMY_NODE_ID,
                                        node: ast::PatIdent(
                                            ast::BindByValue(ast::MutImmutable),
                                                Spanned{
                                                    span: sp(6,7),
                                                    node: str_to_ident("b")},
                                                None
                                                    ),
                                            span: sp(6,7)
                                    }),
                                        id: ast::DUMMY_NODE_ID
                                    }),
                                output: P(ast::Ty{id: ast::DUMMY_NODE_ID,
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
                                        where_clause: ast::WhereClause {
                                            id: ast::DUMMY_NODE_ID,
                                            predicates: Vec::new(),
                                        }
                                    },
                                    P(ast::Block {
                                        view_items: Vec::new(),
                                        stmts: vec!(P(Spanned{
                                            node: ast::StmtSemi(P(ast::Expr{
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
                                                span: sp(17,18)}),
                                                ast::DUMMY_NODE_ID),
                                            span: sp(17,19)})),
                                        expr: None,
                                        id: ast::DUMMY_NODE_ID,
                                        rules: ast::DefaultBlock, // no idea
                                        span: sp(15,21),
                                    })),
                            vis: ast::Inherited,
                            span: sp(0,21)})));
    }


    #[test] fn parse_exprs () {
        // just make sure that they parse....
        string_to_expr("3 + 4".to_string());
        string_to_expr("a::z.froob(b,&(987+3))".to_string());
    }

    #[test] fn attrs_fix_bug () {
        string_to_item("pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<Box<Writer>, String> {
    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
}".to_string());
    }

    #[test] fn crlf_doc_comments() {
        let sess = new_parse_sess();

        let name = "<source>".to_string();
        let source = "/// doc comment\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name.clone(), source, Vec::new(), &sess).unwrap();
        let doc = attr::first_attr_value_str_by_name(item.attrs.as_slice(), "doc").unwrap();
        assert_eq!(doc.get(), "/// doc comment");

        let source = "/// doc comment\r\n/// line 2\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name.clone(), source, Vec::new(), &sess).unwrap();
        let docs = item.attrs.iter().filter(|a| a.name().get() == "doc")
                    .map(|a| a.value_str().unwrap().get().to_string()).collect::<Vec<_>>();
        let b: &[_] = &["/// doc comment".to_string(), "/// line 2".to_string()];
        assert_eq!(docs.as_slice(), b);

        let source = "/** doc comment\r\n *  with CRLF */\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name, source, Vec::new(), &sess).unwrap();
        let doc = attr::first_attr_value_str_by_name(item.attrs.as_slice(), "doc").unwrap();
        assert_eq!(doc.get(), "/** doc comment\n *  with CRLF */");
    }
}
