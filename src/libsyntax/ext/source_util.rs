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
use codemap::{FileMap, Loc, Pos, ExpandedFrom, span};
use codemap::{CallInfo, NameAndSpan};
use ext::base::*;
use ext::base;
use ext::build::{mk_base_vec_e, mk_uint, mk_u8, mk_base_str};
use parse;
use print::pprust;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/* line!(): expands to the current line number */
pub fn expand_line(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = topmost_expn_info(cx.backtrace().get());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);

    base::MRExpr(mk_uint(cx, topmost.call_site, loc.line))
}

/* col!(): expands to the current column number */
pub fn expand_col(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "col!");

    let topmost = topmost_expn_info(cx.backtrace().get());
    let loc = cx.codemap().lookup_char_pos(topmost.call_site.lo);
    base::MRExpr(mk_uint(cx, topmost.call_site, loc.col.to_uint()))
}

/* file!(): expands to the current filename */
/* The filemap (`loc.file`) contains a bunch more information we could spit
 * out if we wanted. */
pub fn expand_file(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = topmost_expn_info(cx.backtrace().get());
    let Loc { file: @FileMap { name: filename, _ }, _ } =
        cx.codemap().lookup_char_pos(topmost.call_site.lo);
    base::MRExpr(mk_base_str(cx, topmost.call_site, filename))
}

pub fn expand_stringify(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let s = pprust::tts_to_str(tts, cx.parse_sess().interner);
    base::MRExpr(mk_base_str(cx, sp, s))
}

pub fn expand_mod(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    base::MRExpr(mk_base_str(cx, sp,
                              str::connect(cx.mod_path().map(
                                  |x| cx.str_of(*x)), ~"::")))
}

// include! : parse the given file as an expr
// This is generally a bad idea because it's going to behave
// unhygienically.
pub fn expand_include(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let file = get_single_str_from_tts(cx, sp, tts, "include!");
    let p = parse::new_sub_parser_from_file(
        cx.parse_sess(), cx.cfg(),
        &res_rel_file(cx, sp, &Path(file)), sp);
    base::MRExpr(p.parse_expr())
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let file = get_single_str_from_tts(cx, sp, tts, "include_str!");
    let res = io::read_whole_file_str(&res_rel_file(cx, sp, &Path(file)));
    match res {
      result::Ok(_) => { /* Continue. */ }
      result::Err(ref e) => {
        cx.parse_sess().span_diagnostic.handler().fatal((*e));
      }
    }

    base::MRExpr(mk_base_str(cx, sp, result::unwrap(res)))
}

pub fn expand_include_bin(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let file = get_single_str_from_tts(cx, sp, tts, "include_bin!");
    match io::read_whole_file(&res_rel_file(cx, sp, &Path(file))) {
      result::Ok(src) => {
        let u8_exprs = vec::map(src, |char| {
            mk_u8(cx, sp, *char)
        });
        base::MRExpr(mk_base_vec_e(cx, sp, u8_exprs))
      }
      result::Err(ref e) => {
        cx.parse_sess().span_diagnostic.handler().fatal((*e))
      }
    }
}

// recur along an ExpnInfo chain to find the original expression
fn topmost_expn_info(expn_info: @codemap::ExpnInfo) -> @codemap::ExpnInfo {
    match *expn_info {
        ExpandedFrom(CallInfo { call_site: ref call_site, _ }) => {
            match call_site.expn_info {
                Some(next_expn_info) => {
                    match *next_expn_info {
                        ExpandedFrom(CallInfo {
                            callee: NameAndSpan { name: ref name, _ },
                            _
                        }) => {
                            // Don't recurse into file using "include!"
                            if *name == ~"include" {
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
fn res_rel_file(cx: @ext_ctxt, sp: codemap::span, arg: &Path) -> Path {
    // NB: relative paths are resolved relative to the compilation unit
    if !arg.is_absolute {
        let cu = Path(cx.codemap().span_to_filename(sp));
        cu.dir_path().push_many(arg.components)
    } else {
        copy *arg
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
