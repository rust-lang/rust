// Copyright 2012-2013 The Rust Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution and at
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

use std::rt::io;
use std::rt::io::extensions::ReaderUtil;
use std::rt::io::file::FileInfo;
use std::str;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/* line!(): expands to the current line number */
pub fn expand_line(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);

    base::MRExpr(cx.expr_uint(topmost.call_site, loc.line))
}

/* col!(): expands to the current column number */
pub fn expand_col(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "col!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);
    base::MRExpr(cx.expr_uint(topmost.call_site, loc.col.to_uint()))
}

/* file!(): expands to the current filename */
/* The filemap (`loc.file`) contains a bunch more information we could spit
 * out if we wanted. */
pub fn expand_file(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = topmost_expn_info(cx.backtrace().unwrap());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);
    let filename = loc.file.name;
    base::MRExpr(cx.expr_str(topmost.call_site, filename))
}

pub fn expand_stringify(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    let s = pprust::tts_to_str(tts, get_ident_interner());
    base::MRExpr(cx.expr_str(sp, s.to_managed()))
}

pub fn expand_mod(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    base::MRExpr(cx.expr_str(sp,
                             cx.mod_path().map(|x| cx.str_of(*x)).connect("::").to_managed()))
}

// include! : parse the given file as an expr
// This is generally a bad idea because it's going to behave
// unhygienically.
pub fn expand_include(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    let file = get_single_str_from_tts(cx, sp, tts, "include!");
    let p = parse::new_sub_parser_from_file(
        cx.parse_sess(), cx.cfg(),
        &res_rel_file(cx, sp, &Path::new(file)), sp);
    base::MRExpr(p.parse_expr())
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    let file = get_single_str_from_tts(cx, sp, tts, "include_str!");
    let file = res_rel_file(cx, sp, &Path::new(file));
    let mut error = None;
    let bytes = do io::io_error::cond.trap(|e| error = Some(e)).inside {
        file.open_reader(io::Open).read_to_end()
    };
    match error {
        Some(e) => {
            cx.span_fatal(sp, format!("couldn't read {}: {}",
                                      file.display(), e.desc));
        }
        None => {}
    }
    match str::from_utf8_owned_opt(bytes) {
        Some(s) => base::MRExpr(cx.expr_str(sp, s.to_managed())),
        None => {
            cx.span_fatal(sp, format!("{} wasn't a utf-8 file", file.display()));
        }
    }
}

pub fn expand_include_bin(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
        -> base::MacResult
{
    use std::at_vec;

    let file = get_single_str_from_tts(cx, sp, tts, "include_bin!");
    let file = res_rel_file(cx, sp, &Path::new(file));

    let mut error = None;
    let bytes = do io::io_error::cond.trap(|e| error = Some(e)).inside {
        file.open_reader(io::Open).read_to_end()
    };
    match error {
        Some(e) => {
            cx.span_fatal(sp, format!("couldn't read {}: {}",
                                      file.display(), e.desc));
        }
        None => {
            let bytes = at_vec::to_managed_move(bytes);
            base::MRExpr(cx.expr_lit(sp, ast::lit_binary(bytes)))
        }
    }
}

// recur along an ExpnInfo chain to find the original expression
fn topmost_expn_info(expn_info: @codemap::ExpnInfo) -> @codemap::ExpnInfo {
    match *expn_info {
        ExpnInfo { call_site: ref call_site, _ } => {
            match call_site.expn_info {
                Some(next_expn_info) => {
                    match *next_expn_info {
                        ExpnInfo {
                            callee: NameAndSpan { name: ref name, _ },
                            _
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
fn res_rel_file(cx: @ExtCtxt, sp: codemap::Span, arg: &Path) -> Path {
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
