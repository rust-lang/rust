// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The compiler code necessary to support the fmt! extension. Eventually this
 * should all get sucked into either the standard library extfmt module or the
 * compiler syntax extension plugin interface.
 */

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;

use std::option;
use std::unstable::extfmt::ct::*;
use parse::token::{str_to_ident};

pub fn expand_syntax_ext(cx: @ExtCtxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let args = get_exprs_from_tts(cx, sp, tts);
    if args.len() == 0 {
        cx.span_fatal(sp, "fmt! takes at least 1 argument.");
    }
    let fmt =
        expr_to_str(cx, args[0],
                    ~"first argument to fmt! must be a string literal.");
    let fmtspan = args[0].span;
    debug!("Format string: %s", fmt);
    fn parse_fmt_err_(cx: @ExtCtxt, sp: span, msg: &str) -> ! {
        cx.span_fatal(sp, msg);
    }
    let parse_fmt_err: @fn(&str) -> ! = |s| parse_fmt_err_(cx, fmtspan, s);
    let pieces = parse_fmt_string(fmt, parse_fmt_err);
    MRExpr(pieces_to_expr(cx, sp, pieces, args))
}

// FIXME (#2249): A lot of these functions for producing expressions can
// probably be factored out in common with other code that builds
// expressions.  Also: Cleanup the naming of these functions.
// Note: Moved many of the common ones to build.rs --kevina
fn pieces_to_expr(cx: @ExtCtxt, sp: span,
                  pieces: ~[Piece], args: ~[@ast::expr])
   -> @ast::expr {
    fn make_path_vec(ident: &str) -> ~[ast::ident] {
        return ~[str_to_ident("std"),
                 str_to_ident("unstable"),
                 str_to_ident("extfmt"),
                 str_to_ident("rt"),
                 str_to_ident(ident)];
    }
    fn make_rt_path_expr(cx: @ExtCtxt, sp: span, nm: &str) -> @ast::expr {
        let path = make_path_vec(nm);
        cx.expr_path(cx.path_global(sp, path))
    }
    // Produces an AST expression that represents a RT::conv record,
    // which tells the RT::conv* functions how to perform the conversion

    fn make_rt_conv_expr(cx: @ExtCtxt, sp: span, cnv: &Conv) -> @ast::expr {
        fn make_flags(cx: @ExtCtxt, sp: span, flags: &[Flag]) -> @ast::expr {
            let mut tmp_expr = make_rt_path_expr(cx, sp, "flag_none");
            foreach f in flags.iter() {
                let fstr = match *f {
                  FlagLeftJustify => "flag_left_justify",
                  FlagLeftZeroPad => "flag_left_zero_pad",
                  FlagSpaceForSign => "flag_space_for_sign",
                  FlagSignAlways => "flag_sign_always",
                  FlagAlternate => "flag_alternate"
                };
                tmp_expr = cx.expr_binary(sp, ast::bitor, tmp_expr,
                                          make_rt_path_expr(cx, sp, fstr));
            }
            return tmp_expr;
        }
        fn make_count(cx: @ExtCtxt, sp: span, cnt: Count) -> @ast::expr {
            match cnt {
              CountImplied => {
                return make_rt_path_expr(cx, sp, "CountImplied");
              }
              CountIs(c) => {
                let count_lit = cx.expr_uint(sp, c as uint);
                let count_is_path = make_path_vec("CountIs");
                let count_is_args = ~[count_lit];
                return cx.expr_call_global(sp, count_is_path, count_is_args);
              }
              _ => cx.span_unimpl(sp, "unimplemented fmt! conversion")
            }
        }
        fn make_ty(cx: @ExtCtxt, sp: span, t: Ty) -> @ast::expr {
            let rt_type = match t {
              TyHex(c) => match c {
                CaseUpper =>  "TyHexUpper",
                CaseLower =>  "TyHexLower"
              },
              TyBits =>  "TyBits",
              TyOctal =>  "TyOctal",
              _ =>  "TyDefault"
            };
            return make_rt_path_expr(cx, sp, rt_type);
        }
        fn make_conv_struct(cx: @ExtCtxt, sp: span, flags_expr: @ast::expr,
                         width_expr: @ast::expr, precision_expr: @ast::expr,
                         ty_expr: @ast::expr) -> @ast::expr {
            cx.expr_struct(
                sp,
                cx.path_global(sp, make_path_vec("Conv")),
                ~[
                    cx.field_imm(sp, str_to_ident("flags"), flags_expr),
                    cx.field_imm(sp, str_to_ident("width"), width_expr),
                    cx.field_imm(sp, str_to_ident("precision"), precision_expr),
                    cx.field_imm(sp, str_to_ident("ty"), ty_expr)
                ]
            )
        }
        let rt_conv_flags = make_flags(cx, sp, cnv.flags);
        let rt_conv_width = make_count(cx, sp, cnv.width);
        let rt_conv_precision = make_count(cx, sp, cnv.precision);
        let rt_conv_ty = make_ty(cx, sp, cnv.ty);
        make_conv_struct(cx, sp, rt_conv_flags, rt_conv_width,
                         rt_conv_precision, rt_conv_ty)
    }
    fn make_conv_call(cx: @ExtCtxt, sp: span, conv_type: &str, cnv: &Conv,
                      arg: @ast::expr, buf: @ast::expr) -> @ast::expr {
        let fname = ~"conv_" + conv_type;
        let path = make_path_vec(fname);
        let cnv_expr = make_rt_conv_expr(cx, sp, cnv);
        let args = ~[cnv_expr, arg, buf];
        cx.expr_call_global(arg.span, path, args)
    }

    fn make_new_conv(cx: @ExtCtxt, sp: span, cnv: &Conv,
                     arg: @ast::expr, buf: @ast::expr) -> @ast::expr {
        fn is_signed_type(cnv: &Conv) -> bool {
            match cnv.ty {
              TyInt(s) => match s {
                Signed => return true,
                Unsigned => return false
              },
              TyFloat => return true,
              _ => return false
            }
        }
        let unsupported = ~"conversion not supported in fmt! string";
        match cnv.param {
          option::None => (),
          _ => cx.span_unimpl(sp, unsupported)
        }
        foreach f in cnv.flags.iter() {
            match *f {
              FlagLeftJustify => (),
              FlagSignAlways => {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  "+ flag only valid in \
                                   signed fmt! conversion");
                }
              }
              FlagSpaceForSign => {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  "space flag only valid in \
                                   signed fmt! conversions");
                }
              }
              FlagLeftZeroPad => (),
              _ => cx.span_unimpl(sp, unsupported)
            }
        }
        match cnv.width {
          CountImplied => (),
          CountIs(_) => (),
          _ => cx.span_unimpl(sp, unsupported)
        }
        match cnv.precision {
          CountImplied => (),
          CountIs(_) => (),
          _ => cx.span_unimpl(sp, unsupported)
        }
        let (name, actual_arg) = match cnv.ty {
            TyStr => ("str", arg),
            TyInt(Signed) => ("int", arg),
            TyBool => ("bool", arg),
            TyChar => ("char", arg),
            TyBits | TyOctal | TyHex(_) | TyInt(Unsigned) => ("uint", arg),
            TyFloat => ("float", arg),
            TyPointer => ("pointer", arg),
            TyPoly => ("poly", cx.expr_addr_of(sp, arg))
        };
        return make_conv_call(cx, arg.span, name, cnv, actual_arg,
                              cx.expr_mut_addr_of(arg.span, buf));
    }
    fn log_conv(c: &Conv) {
        debug!("Building conversion:");
        match c.param {
          Some(p) => { debug!("param: %s", p.to_str()); }
          _ => debug!("param: none")
        }
        foreach f in c.flags.iter() {
            match *f {
              FlagLeftJustify => debug!("flag: left justify"),
              FlagLeftZeroPad => debug!("flag: left zero pad"),
              FlagSpaceForSign => debug!("flag: left space pad"),
              FlagSignAlways => debug!("flag: sign always"),
              FlagAlternate => debug!("flag: alternate")
            }
        }
        match c.width {
          CountIs(i) =>
              debug!("width: count is %s", i.to_str()),
          CountIsParam(i) =>
              debug!("width: count is param %s", i.to_str()),
          CountIsNextParam => debug!("width: count is next param"),
          CountImplied => debug!("width: count is implied")
        }
        match c.precision {
          CountIs(i) =>
              debug!("prec: count is %s", i.to_str()),
          CountIsParam(i) =>
              debug!("prec: count is param %s", i.to_str()),
          CountIsNextParam => debug!("prec: count is next param"),
          CountImplied => debug!("prec: count is implied")
        }
        match c.ty {
          TyBool => debug!("type: bool"),
          TyStr => debug!("type: str"),
          TyChar => debug!("type: char"),
          TyInt(s) => match s {
            Signed => debug!("type: signed"),
            Unsigned => debug!("type: unsigned")
          },
          TyBits => debug!("type: bits"),
          TyHex(cs) => match cs {
            CaseUpper => debug!("type: uhex"),
            CaseLower => debug!("type: lhex"),
          },
          TyOctal => debug!("type: octal"),
          TyFloat => debug!("type: float"),
          TyPointer => debug!("type: pointer"),
          TyPoly => debug!("type: poly")
        }
    }

    /* Short circuit an easy case up front (won't work otherwise) */
    if pieces.len() == 0 {
        return cx.expr_str_uniq(args[0].span, @"");
    }

    let fmt_sp = args[0].span;
    let mut n = 0u;
    let nargs = args.len();

    /* 'ident' is the local buffer building up the result of fmt! */
    let ident = str_to_ident("__fmtbuf");
    let buf = || cx.expr_ident(fmt_sp, ident);
    let core_ident = str_to_ident("std");
    let str_ident = str_to_ident("str");
    let push_ident = str_to_ident("push_str");
    let mut stms = ~[];

    /* Translate each piece (portion of the fmt expression) by invoking the
       corresponding function in std::unstable::extfmt. Each function takes a
       buffer to insert data into along with the data being formatted. */
    let npieces = pieces.len();
    foreach (i, pc) in pieces.consume_iter().enumerate() {
        match pc {
            /* Raw strings get appended via str::push_str */
            PieceString(s) => {
                /* If this is the first portion, then initialize the local
                   buffer with it directly. If it's actually the only piece,
                   then there's no need for it to be mutable */
                if i == 0 {
                    stms.push(cx.stmt_let(fmt_sp, npieces > 1,
                                          ident, cx.expr_str_uniq(fmt_sp, s.to_managed())));
                } else {
                    // we call the push_str function because the
                    // bootstrap doesnt't seem to work if we call the
                    // method.
                    let args = ~[cx.expr_mut_addr_of(fmt_sp, buf()),
                                 cx.expr_str(fmt_sp, s.to_managed())];
                    let call = cx.expr_call_global(fmt_sp,
                                                   ~[core_ident,
                                                     str_ident,
                                                     push_ident],
                                                   args);
                    stms.push(cx.stmt_expr(call));
                }
            }

            /* Invoke the correct conv function in extfmt */
            PieceConv(ref conv) => {
                n += 1u;
                if n >= nargs {
                    cx.span_fatal(sp,
                                  "not enough arguments to fmt! \
                                   for the given format string");
                }

                log_conv(conv);
                /* If the first portion is a conversion, then the local buffer
                   must be initialized as an empty string */
                if i == 0 {
                    stms.push(cx.stmt_let(fmt_sp, true, ident,
                                          cx.expr_str_uniq(fmt_sp, @"")));
                }
                stms.push(cx.stmt_expr(make_new_conv(cx, fmt_sp, conv,
                                                     args[n], buf())));
            }
        }
    }

    let expected_nargs = n + 1u; // n conversions + the fmt string
    if expected_nargs < nargs {
        cx.span_fatal
            (sp, fmt!("too many arguments to fmt!. found %u, expected %u",
                           nargs, expected_nargs));
    }

    cx.expr_block(cx.block(fmt_sp, stms, Some(buf())))
}
