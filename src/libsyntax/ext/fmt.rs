

/*
 * The compiler code necessary to support the #fmt extension. Eventually this
 * should all get sucked into either the standard library extfmt module or the
 * compiler syntax extension plugin interface.
 */
use extfmt::ct::*;
use base::*;
use codemap::span;
use ext::build::*;
export expand_syntax_ext;

fn expand_syntax_ext(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args_no_max(cx, sp, arg, 1u, ~"fmt");
    let fmt =
        expr_to_str(cx, args[0],
                    ~"first argument to #fmt must be a string literal.");
    let fmtspan = args[0].span;
    debug!("Format string:");
    log(debug, fmt);
    fn parse_fmt_err_(cx: ext_ctxt, sp: span, msg: ~str) -> ! {
        cx.span_fatal(sp, msg);
    }
    let parse_fmt_err = fn@(s: ~str) -> ! {
        parse_fmt_err_(cx, fmtspan, s)
    };
    let pieces = parse_fmt_string(fmt, parse_fmt_err);
    return pieces_to_expr(cx, sp, pieces, args);
}

// FIXME (#2249): A lot of these functions for producing expressions can
// probably be factored out in common with other code that builds
// expressions.  Also: Cleanup the naming of these functions.
// NOTE: Moved many of the common ones to build.rs --kevina
fn pieces_to_expr(cx: ext_ctxt, sp: span,
                  pieces: ~[Piece], args: ~[@ast::expr])
   -> @ast::expr {
    fn make_path_vec(_cx: ext_ctxt, ident: @~str) -> ~[ast::ident] {
        let intr = _cx.parse_sess().interner;
        return ~[intr.intern(@~"extfmt"), intr.intern(@~"rt"),
                 intr.intern(ident)];
    }
    fn make_rt_path_expr(cx: ext_ctxt, sp: span, nm: @~str) -> @ast::expr {
        let path = make_path_vec(cx, nm);
        return mk_path(cx, sp, path);
    }
    // Produces an AST expression that represents a RT::conv record,
    // which tells the RT::conv* functions how to perform the conversion

    fn make_rt_conv_expr(cx: ext_ctxt, sp: span, cnv: Conv) -> @ast::expr {
        fn make_flags(cx: ext_ctxt, sp: span, flags: ~[Flag]) -> @ast::expr {
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
        fn make_count(cx: ext_ctxt, sp: span, cnt: Count) -> @ast::expr {
            match cnt {
              CountImplied => {
                return make_rt_path_expr(cx, sp, @~"CountImplied");
              }
              CountIs(c) => {
                let count_lit = mk_int(cx, sp, c);
                let count_is_path = make_path_vec(cx, @~"CountIs");
                let count_is_args = ~[count_lit];
                return mk_call(cx, sp, count_is_path, count_is_args);
              }
              _ => cx.span_unimpl(sp, ~"unimplemented #fmt conversion")
            }
        }
        fn make_ty(cx: ext_ctxt, sp: span, t: Ty) -> @ast::expr {
            let mut rt_type;
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
        fn make_conv_rec(cx: ext_ctxt, sp: span, flags_expr: @ast::expr,
                         width_expr: @ast::expr, precision_expr: @ast::expr,
                         ty_expr: @ast::expr) -> @ast::expr {
            let intr = cx.parse_sess().interner;
            return mk_rec_e(cx, sp,
                         ~[{ident: intr.intern(@~"flags"), ex: flags_expr},
                           {ident: intr.intern(@~"width"), ex: width_expr},
                           {ident: intr.intern(@~"precision"),
                            ex: precision_expr},
                           {ident: intr.intern(@~"ty"), ex: ty_expr}]);
        }
        let rt_conv_flags = make_flags(cx, sp, cnv.flags);
        let rt_conv_width = make_count(cx, sp, cnv.width);
        let rt_conv_precision = make_count(cx, sp, cnv.precision);
        let rt_conv_ty = make_ty(cx, sp, cnv.ty);
        return make_conv_rec(cx, sp, rt_conv_flags, rt_conv_width,
                          rt_conv_precision, rt_conv_ty);
    }
    fn make_conv_call(cx: ext_ctxt, sp: span, conv_type: ~str, cnv: Conv,
                      arg: @ast::expr) -> @ast::expr {
        let fname = ~"conv_" + conv_type;
        let path = make_path_vec(cx, @fname);
        let cnv_expr = make_rt_conv_expr(cx, sp, cnv);
        let args = ~[cnv_expr, arg];
        return mk_call(cx, arg.span, path, args);
    }

    fn make_new_conv(cx: ext_ctxt, sp: span, cnv: Conv, arg: @ast::expr) ->
       @ast::expr {
        // FIXME: Move validation code into core::extfmt (Issue #2249)

        fn is_signed_type(cnv: Conv) -> bool {
            match cnv.ty {
              TyInt(s) => match s {
                Signed => return true,
                Unsigned => return false
              },
              TyFloat => return true,
              _ => return false
            }
        }
        let unsupported = ~"conversion not supported in #fmt string";
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
                                      ~"signed #fmt conversion");
                }
              }
              FlagSpaceForSign => {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  ~"space flag only valid in " +
                                      ~"signed #fmt conversions");
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
        match cnv.ty {
          TyStr => return make_conv_call(cx, arg.span, ~"str", cnv, arg),
          TyInt(sign) => match sign {
            Signed => return make_conv_call(cx, arg.span, ~"int", cnv, arg),
            Unsigned => {
                return make_conv_call(cx, arg.span, ~"uint", cnv, arg)
            }
          },
          TyBool => return make_conv_call(cx, arg.span, ~"bool", cnv, arg),
          TyChar => return make_conv_call(cx, arg.span, ~"char", cnv, arg),
          TyHex(_) => {
            return make_conv_call(cx, arg.span, ~"uint", cnv, arg);
          }
          TyBits => return make_conv_call(cx, arg.span, ~"uint", cnv, arg),
          TyOctal => return make_conv_call(cx, arg.span, ~"uint", cnv, arg),
          TyFloat => {
            return make_conv_call(cx, arg.span, ~"float", cnv, arg);
          }
          TyPoly => return make_conv_call(cx, arg.span, ~"poly", cnv, arg)
        }
    }
    fn log_conv(c: Conv) {
        match c.param {
          Some(p) => { log(debug, ~"param: " + int::to_str(p, 10u)); }
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
          CountIs(i) => log(
              debug, ~"width: count is " + int::to_str(i, 10u)),
          CountIsParam(i) => log(
              debug, ~"width: count is param " + int::to_str(i, 10u)),
          CountIsNextParam => debug!("width: count is next param"),
          CountImplied => debug!("width: count is implied")
        }
        match c.precision {
          CountIs(i) => log(
              debug, ~"prec: count is " + int::to_str(i, 10u)),
          CountIsParam(i) => log(
              debug, ~"prec: count is param " + int::to_str(i, 10u)),
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
    let mut piece_exprs = ~[];
    let nargs = args.len();
    for pieces.each |pc| {
        match *pc {
          PieceString(s) => {
            vec::push(piece_exprs, mk_uniq_str(cx, fmt_sp, s))
          }
          PieceConv(conv) => {
            n += 1u;
            if n >= nargs {
                cx.span_fatal(sp,
                              ~"not enough arguments to #fmt " +
                                  ~"for the given format string");
            }
            debug!("Building conversion:");
            log_conv(conv);
            let arg_expr = args[n];
            let c_expr = make_new_conv(cx, fmt_sp, conv, arg_expr);
            vec::push(piece_exprs, c_expr);
          }
        }
    }
    let expected_nargs = n + 1u; // n conversions + the fmt string

    if expected_nargs < nargs {
        cx.span_fatal
            (sp, fmt!("too many arguments to #fmt. found %u, expected %u",
                           nargs, expected_nargs));
    }

    let arg_vec = mk_fixed_vec_e(cx, fmt_sp, piece_exprs);
    return mk_call(cx, fmt_sp,
                   ~[cx.parse_sess().interner.intern(@~"str"),
                     cx.parse_sess().interner.intern(@~"concat")],
                   ~[arg_vec]);
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
