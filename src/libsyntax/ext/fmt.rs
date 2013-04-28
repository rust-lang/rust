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
use ext::build;
use ext::build::*;

use core::unstable::extfmt::ct::*;

pub fn expand_syntax_ext(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let args = get_exprs_from_tts(cx, tts);
    if args.len() == 0 {
        cx.span_fatal(sp, "fmt! takes at least 1 argument.");
    }
    let fmt =
        expr_to_str(cx, args[0],
                    ~"first argument to fmt! must be a string literal.");
    let fmtspan = args[0].span;
    debug!("Format string: %s", fmt);
    fn parse_fmt_err_(cx: @ext_ctxt, sp: span, msg: &str) -> ! {
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
fn pieces_to_expr(cx: @ext_ctxt, sp: span,
                  pieces: ~[Piece], args: ~[@ast::expr])
   -> @ast::expr {
    fn make_path_vec(cx: @ext_ctxt, ident: @~str) -> ~[ast::ident] {
        let intr = cx.parse_sess().interner;
        return ~[intr.intern(@~"unstable"), intr.intern(@~"extfmt"),
                 intr.intern(@~"rt"), intr.intern(ident)];
    }
    fn make_rt_path_expr(cx: @ext_ctxt, sp: span, nm: @~str) -> @ast::expr {
        let path = make_path_vec(cx, nm);
        return mk_path_global(cx, sp, path);
    }
    // Produces an AST expression that represents a RT::conv record,
    // which tells the RT::conv* functions how to perform the conversion

    fn make_rt_conv_expr(cx: @ext_ctxt, sp: span, cnv: &Conv) -> @ast::expr {
        fn make_flags(cx: @ext_ctxt, sp: span, flags: ~[Flag]) -> @ast::expr {
            let mut tmp_expr = make_rt_path_expr(cx, sp, @~"flag_none");
            for flags.each |f| {
                let fstr = match *f {
                  FlagLeftJustify => ~"flag_left_justify",
                  FlagLeftZeroPad => ~"flag_left_zero_pad",
                  FlagSpaceForSign => ~"flag_space_for_sign",
                  FlagSignAlways => ~"flag_sign_always",
                  FlagAlternate => ~"flag_alternate"
                };
                tmp_expr = mk_binary(cx, sp, ast::bitor, tmp_expr,
                                     make_rt_path_expr(cx, sp, @fstr));
            }
            return tmp_expr;
        }
        fn make_count(cx: @ext_ctxt, sp: span, cnt: Count) -> @ast::expr {
            match cnt {
              CountImplied => {
                return make_rt_path_expr(cx, sp, @~"CountImplied");
              }
              CountIs(c) => {
                let count_lit = mk_uint(cx, sp, c as uint);
                let count_is_path = make_path_vec(cx, @~"CountIs");
                let count_is_args = ~[count_lit];
                return mk_call_global(cx, sp, count_is_path, count_is_args);
              }
              _ => cx.span_unimpl(sp, ~"unimplemented fmt! conversion")
            }
        }
        fn make_ty(cx: @ext_ctxt, sp: span, t: Ty) -> @ast::expr {
            let rt_type;
            match t {
              TyHex(c) => match c {
                CaseUpper => rt_type = ~"TyHexUpper",
                CaseLower => rt_type = ~"TyHexLower"
              },
              TyBits => rt_type = ~"TyBits",
              TyOctal => rt_type = ~"TyOctal",
              _ => rt_type = ~"TyDefault"
            }
            return make_rt_path_expr(cx, sp, @rt_type);
        }
        fn make_conv_struct(cx: @ext_ctxt, sp: span, flags_expr: @ast::expr,
                         width_expr: @ast::expr, precision_expr: @ast::expr,
                         ty_expr: @ast::expr) -> @ast::expr {
            let intr = cx.parse_sess().interner;
            mk_global_struct_e(
                cx,
                sp,
                make_path_vec(cx, @~"Conv"),
                ~[
                    build::Field {
                        ident: intr.intern(@~"flags"), ex: flags_expr
                    },
                    build::Field {
                        ident: intr.intern(@~"width"), ex: width_expr
                    },
                    build::Field {
                        ident: intr.intern(@~"precision"), ex: precision_expr
                    },
                    build::Field {
                        ident: intr.intern(@~"ty"), ex: ty_expr
                    },
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
    fn make_conv_call(cx: @ext_ctxt, sp: span, conv_type: &str, cnv: &Conv,
                      arg: @ast::expr, buf: @ast::expr) -> @ast::expr {
        let fname = ~"conv_" + conv_type;
        let path = make_path_vec(cx, @fname);
        let cnv_expr = make_rt_conv_expr(cx, sp, cnv);
        let args = ~[cnv_expr, arg, buf];
        return mk_call_global(cx, arg.span, path, args);
    }

    fn make_new_conv(cx: @ext_ctxt, sp: span, cnv: &Conv,
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
        for cnv.flags.each |f| {
            match *f {
              FlagLeftJustify => (),
              FlagSignAlways => {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  ~"+ flag only valid in " +
                                      ~"signed fmt! conversion");
                }
              }
              FlagSpaceForSign => {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  ~"space flag only valid in " +
                                      ~"signed fmt! conversions");
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
            TyPoly => ("poly", mk_addr_of(cx, sp, arg))
        };
        return make_conv_call(cx, arg.span, name, cnv, actual_arg,
                              mk_mut_addr_of(cx, arg.span, buf));
    }
    fn log_conv(c: &Conv) {
        debug!("Building conversion:");
        match c.param {
          Some(p) => { debug!("param: %s", p.to_str()); }
          _ => debug!("param: none")
        }
        for c.flags.each |f| {
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
          TyPoly => debug!("type: poly")
        }
    }

    let fmt_sp = args[0].span;
    let mut n = 0u;
    let nargs = args.len();

    /* 'ident' is the local buffer building up the result of fmt! */
    let ident = cx.parse_sess().interner.intern(@~"__fmtbuf");
    let buf = || mk_path(cx, fmt_sp, ~[ident]);
    let str_ident = cx.parse_sess().interner.intern(@~"str");
    let push_ident = cx.parse_sess().interner.intern(@~"push_str");
    let mut stms = ~[];

    /* Translate each piece (portion of the fmt expression) by invoking the
       corresponding function in core::unstable::extfmt. Each function takes a
       buffer to insert data into along with the data being formatted. */
    let npieces = pieces.len();
    do vec::consume(pieces) |i, pc| {
        match pc {
            /* Raw strings get appended via str::push_str */
            PieceString(s) => {
                let portion = mk_uniq_str(cx, fmt_sp, s);

                /* If this is the first portion, then initialize the local
                   buffer with it directly. If it's actually the only piece,
                   then there's no need for it to be mutable */
                if i == 0 {
                    stms.push(mk_local(cx, fmt_sp, npieces > 1, ident, portion));
                } else {
                    let args = ~[mk_mut_addr_of(cx, fmt_sp, buf()), portion];
                    let call = mk_call_global(cx,
                                              fmt_sp,
                                              ~[str_ident, push_ident],
                                              args);
                    stms.push(mk_stmt(cx, fmt_sp, call));
                }
            }

            /* Invoke the correct conv function in extfmt */
            PieceConv(ref conv) => {
                n += 1u;
                if n >= nargs {
                    cx.span_fatal(sp,
                                  ~"not enough arguments to fmt! " +
                                  ~"for the given format string");
                }

                log_conv(conv);
                /* If the first portion is a conversion, then the local buffer
                   must be initialized as an empty string */
                if i == 0 {
                    stms.push(mk_local(cx, fmt_sp, true, ident,
                                       mk_uniq_str(cx, fmt_sp, ~"")));
                }
                stms.push(mk_stmt(cx, fmt_sp,
                                  make_new_conv(cx, fmt_sp, conv,
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

    return mk_block(cx, fmt_sp, ~[], stms, Some(buf()));
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
