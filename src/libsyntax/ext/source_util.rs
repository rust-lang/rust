// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use syntax_pos::{self, Pos, Span, FileName};
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use parse::{token, DirectoryOwnership};
use parse;
use print::pprust;
use ptr::P;
use OneVector;
use symbol::Symbol;
use tokenstream;

use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use rustc_data_structures::sync::Lrc;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/// line!(): expands to the current line number
pub fn expand_line(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                   -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.line as u32))
}

/* column!(): expands to the current column number */
pub fn expand_column(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "column!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.col.to_usize() as u32 + 1))
}

/* __rust_unstable_column!(): expands to the current column number */
pub fn expand_column_gated(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    if sp.allows_unstable() {
        expand_column(cx, sp, tts)
    } else {
        cx.span_fatal(sp, "the __rust_unstable_column macro is unstable");
    }
}

/// file!(): expands to the current filename */
/// The source_file (`loc.file`) contains a bunch more information we could spit
/// out if we wanted.
pub fn expand_file(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                   -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());
    base::MacEager::expr(cx.expr_str(topmost, Symbol::intern(&loc.file.name.to_string())))
}

pub fn expand_stringify(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let s = pprust::tts_to_string(tts);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&s)))
}

pub fn expand_mod(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    let mod_path = &cx.current_expansion.module.mod_path;
    let string = mod_path.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("::");

    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&string)))
}

/// include! : parse the given file as an expr
/// This is generally a bad idea because it's going to behave
/// unhygienically.
pub fn expand_include<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                           -> Box<dyn base::MacResult+'cx> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Some(f) => f,
        None => return DummyResult::expr(sp),
    };
    // The file will be added to the code map by the parser
    let path = res_rel_file(cx, sp, file);
    let directory_ownership = DirectoryOwnership::Owned { relative: None };
    let p = parse::new_sub_parser_from_file(cx.parse_sess(), &path, directory_ownership, None, sp);

    struct ExpandResult<'a> {
        p: parse::parser::Parser<'a>,
    }
    impl<'a> base::MacResult for ExpandResult<'a> {
        fn make_expr(mut self: Box<ExpandResult<'a>>) -> Option<P<ast::Expr>> {
            Some(panictry!(self.p.parse_expr()))
        }
        fn make_items(mut self: Box<ExpandResult<'a>>)
                      -> Option<OneVector<P<ast::Item>>> {
            let mut ret = OneVector::new();
            while self.p.token != token::Eof {
                match panictry!(self.p.parse_item()) {
                    Some(item) => ret.push(item),
                    None => self.p.diagnostic().span_fatal(self.p.span,
                                                           &format!("expected item, found `{}`",
                                                                    self.p.this_token_to_string()))
                                               .raise()
                }
            }
            Some(ret)
        }
    }

    Box::new(ExpandResult { p: p })
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                          -> Box<dyn base::MacResult+'static> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include_str!") {
        Some(f) => f,
        None => return DummyResult::expr(sp)
    };
    let file = res_rel_file(cx, sp, file);
    let mut bytes = Vec::new();
    match File::open(&file).and_then(|mut f| f.read_to_end(&mut bytes)) {
        Ok(..) => {}
        Err(e) => {
            cx.span_err(sp,
                        &format!("couldn't read {}: {}",
                                file.display(),
                                e));
            return DummyResult::expr(sp);
        }
    };
    match String::from_utf8(bytes) {
        Ok(src) => {
            let interned_src = Symbol::intern(&src);

            // Add this input file to the code map to make it available as
            // dependency information
            cx.source_map().new_source_file(file.into(), src);

            base::MacEager::expr(cx.expr_str(sp, interned_src))
        }
        Err(_) => {
            cx.span_err(sp,
                        &format!("{} wasn't a utf-8 file",
                                file.display()));
            DummyResult::expr(sp)
        }
    }
}

pub fn expand_include_bytes(cx: &mut ExtCtxt, sp: Span, tts: &[tokenstream::TokenTree])
                            -> Box<dyn base::MacResult+'static> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include_bytes!") {
        Some(f) => f,
        None => return DummyResult::expr(sp)
    };
    let file = res_rel_file(cx, sp, file);
    let mut bytes = Vec::new();
    match File::open(&file).and_then(|mut f| f.read_to_end(&mut bytes)) {
        Err(e) => {
            cx.span_err(sp,
                        &format!("couldn't read {}: {}", file.display(), e));
            DummyResult::expr(sp)
        }
        Ok(..) => {
            // Add this input file to the code map to make it available as
            // dependency information, but don't enter it's contents
            cx.source_map().new_source_file(file.into(), "".to_string());

            base::MacEager::expr(cx.expr_lit(sp, ast::LitKind::ByteStr(Lrc::new(bytes))))
        }
    }
}

// resolve a file-system path to an absolute file-system path (if it
// isn't already)
fn res_rel_file(cx: &mut ExtCtxt, sp: syntax_pos::Span, arg: String) -> PathBuf {
    let arg = PathBuf::from(arg);
    // Relative paths are resolved relative to the file in which they are found
    // after macro expansion (that is, they are unhygienic).
    if !arg.is_absolute() {
        let callsite = sp.source_callsite();
        let mut path = match cx.source_map().span_to_unmapped_path(callsite) {
            FileName::Real(path) => path,
            other => panic!("cannot resolve relative path in non-file source `{}`", other),
        };
        path.pop();
        path.push(arg);
        path
    } else {
        arg
    }
}
