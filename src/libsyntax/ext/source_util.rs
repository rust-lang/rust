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
use codemap;
use codemap::{Pos, Span};
use codemap::{ExpnInfo, NameAndSpan};
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use parse;
use parse::token;
use print::pprust;

use std::io::File;
use std::rc::Rc;
use std::str;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/* line!(): expands to the current line number */
pub fn expand_line(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);

    base::MRExpr(cx.expr_uint(topmost.call_site, loc.line))
}

/* col!(): expands to the current column number */
pub fn expand_col(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "col!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);
    base::MRExpr(cx.expr_uint(topmost.call_site, loc.col.to_uint()))
}

/* file!(): expands to the current filename */
/* The filemap (`loc.file`) contains a bunch more information we could spit
 * out if we wanted. */
pub fn expand_file(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);
    let filename = token::intern_and_get_ident(loc.file.deref().name);
    base::MRExpr(cx.expr_str(topmost.call_site, filename))
}

pub fn expand_stringify(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let s = pprust::tts_to_str(tts);
    base::MRExpr(cx.expr_str(sp, token::intern_and_get_ident(s)))
}

pub fn expand_mod(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    let string = cx.mod_path()
                   .map(|x| token::get_ident(*x).get().to_str())
                   .connect("::");
    base::MRExpr(cx.expr_str(sp, token::intern_and_get_ident(string)))
}

// include! : parse the given file as an expr
// This is generally a bad idea because it's going to behave
// unhygienically.
pub fn expand_include(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Some(f) => f,
        None => return MacResult::dummy_expr(sp),
    };
    // The file will be added to the code map by the parser
    let mut p =
        parse::new_sub_parser_from_file(cx.parse_sess(),
                                        cx.cfg(),
                                        &res_rel_file(cx,
                                                      sp,
                                                      &Path::new(file)),
                                        sp);
    base::MRExpr(p.parse_expr())
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let file = match get_single_str_from_tts(cx, sp, tts, "include_str!") {
        Some(f) => f,
        None => return MacResult::dummy_expr(sp)
    };
    let file = res_rel_file(cx, sp, &Path::new(file));
    let bytes = match File::open(&file).read_to_end() {
        Err(e) => {
            cx.span_err(sp, format!("couldn't read {}: {}", file.display(), e));
            return MacResult::dummy_expr(sp);
        }
        Ok(bytes) => bytes,
    };
    match str::from_utf8_owned(bytes) {
        Some(src) => {
            // Add this input file to the code map to make it available as
            // dependency information
            let filename = file.display().to_str();
            let interned = token::intern_and_get_ident(src);
            cx.codemap().new_filemap(filename, src);

            base::MRExpr(cx.expr_str(sp, interned))
        }
        None => {
            cx.span_err(sp, format!("{} wasn't a utf-8 file", file.display()));
            return MacResult::dummy_expr(sp);
        }
    }
}

pub fn expand_include_bin(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
        -> base::MacResult
{
    let file = match get_single_str_from_tts(cx, sp, tts, "include_bin!") {
        Some(f) => f,
        None => return MacResult::dummy_expr(sp)
    };
    let file = res_rel_file(cx, sp, &Path::new(file));
    match File::open(&file).read_to_end() {
        Err(e) => {
            cx.span_err(sp, format!("couldn't read {}: {}", file.display(), e));
            return MacResult::dummy_expr(sp);
        }
        Ok(bytes) => {
            let bytes = bytes.iter().map(|x| *x).collect();
            base::MRExpr(cx.expr_lit(sp, ast::LitBinary(Rc::new(bytes))))
        }
    }
}

// recur along an ExpnInfo chain to find the original expression
fn topmost_expn_info(expn_info: @codemap::ExpnInfo) -> @codemap::ExpnInfo {
    match *expn_info {
        ExpnInfo { call_site: ref call_site, .. } => {
            match call_site.expn_info {
                Some(next_expn_info) => {
                    match *next_expn_info {
                        ExpnInfo {
                            callee: NameAndSpan { name: ref name, .. },
                            ..
                        } => {
                            // Don't recurse into file using "include!"
                            if "include" == *name  {
                                expn_info
                            } else {
                                topmost_expn_info(next_expn_info)
                            }
                        }
                    }
                },
                None => expn_info
            }
        }
    }
}

// resolve a file-system path to an absolute file-system path (if it
// isn't already)
fn res_rel_file(cx: &mut ExtCtxt, sp: codemap::Span, arg: &Path) -> Path {
    // NB: relative paths are resolved relative to the compilation unit
    if !arg.is_absolute() {
        let mut cu = Path::new(cx.codemap().span_to_filename(sp));
        cu.pop();
        cu.push(arg);
        cu
    } else {
        arg.clone()
    }
}
