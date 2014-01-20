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
use parse::token::{get_ident_interner};
use print::pprust;

use std::io;
use std::io::File;
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
    let filename = loc.file.name;
    base::MRExpr(cx.expr_str(topmost.call_site, filename))
}

pub fn expand_stringify(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let s = pprust::tts_to_str(tts, get_ident_interner());
    base::MRExpr(cx.expr_str(sp, s.to_managed()))
}

pub fn expand_mod(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    base::MRExpr(cx.expr_str(sp,
                             cx.mod_path().map(|x| cx.str_of(*x)).connect("::").to_managed()))
}

// include! : parse the given file as an expr
// This is generally a bad idea because it's going to behave
// unhygienically.
pub fn expand_include(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Some(f) => f,
        None => return MacResult::dummy_expr(),
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
        None => return MacResult::dummy_expr()
    };
    let file = res_rel_file(cx, sp, &Path::new(file));
    let bytes = match io::result(|| File::open(&file).read_to_end()) {
        Err(e) => {
            cx.span_err(sp, format!("couldn't read {}: {}", file.display(), e.desc));
            return MacResult::dummy_expr();
        }
        Ok(bytes) => bytes,
    };
    match str::from_utf8_owned_opt(bytes) {
        Some(src) => {
            // Add this input file to the code map to make it available as
            // dependency information
            let src = src.to_managed();
            let filename = file.display().to_str().to_managed();
            cx.parse_sess.cm.new_filemap(filename, src);

            base::MRExpr(cx.expr_str(sp, src))
        }
        None => {
            cx.span_err(sp, format!("{} wasn't a utf-8 file", file.display()));
            return MacResult::dummy_expr();
        }
    }
}

pub fn expand_include_bin(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
        -> base::MacResult
{
    use std::at_vec;

    let file = match get_single_str_from_tts(cx, sp, tts, "include_bin!") {
        Some(f) => f,
        None => return MacResult::dummy_expr()
    };
    let file = res_rel_file(cx, sp, &Path::new(file));
    match io::result(|| File::open(&file).read_to_end()) {
        Err(e) => {
            cx.span_err(sp, format!("couldn't read {}: {}", file.display(), e.desc));
            return MacResult::dummy_expr();
        }
        Ok(bytes) => {
            let bytes = at_vec::to_managed_move(bytes);
            base::MRExpr(cx.expr_lit(sp, ast::LitBinary(bytes)))
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
