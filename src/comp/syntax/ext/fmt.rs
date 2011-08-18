

/*
 * The compiler code necessary to support the #fmt extension.  Eventually this
 * should all get sucked into either the standard library extfmt module or the
 * compiler syntax extension plugin interface.
 */
import std::vec;
import std::str;
import std::option;
import std::option::none;
import std::option::some;
import std::extfmt::ct::*;
import base::*;
import codemap::span;
export expand_syntax_ext;

fn expand_syntax_ext(cx: &ext_ctxt, sp: span, arg: @ast::expr,
                     _body: option::t<str>) -> @ast::expr {
    let args: [@ast::expr] = alt arg.node {
      ast::expr_vec(elts, _) { elts }
      _ { cx.span_fatal(sp, "#fmt requires arguments of the form `[...]`.") }
    };
    if vec::len::<@ast::expr>(args) == 0u {
        cx.span_fatal(sp, "#fmt requires a format string");
    }
    let fmt =
        expr_to_str(cx, args.(0),
                    "first argument to #fmt must be a " + "string literal.");
    let fmtspan = args.(0).span;
    log "Format string:";
    log fmt;
    fn parse_fmt_err_(cx: &ext_ctxt, sp: span, msg: str) -> ! {
        cx.span_fatal(sp, msg);
    }
    let parse_fmt_err = bind parse_fmt_err_(cx, fmtspan, _);
    let pieces = parse_fmt_string(fmt, parse_fmt_err);
    ret pieces_to_expr(cx, sp, pieces, args);
}

// FIXME: A lot of these functions for producing expressions can probably
// be factored out in common with other code that builds expressions.
// FIXME: Cleanup the naming of these functions
fn pieces_to_expr(cx: &ext_ctxt, sp: span, pieces: &[piece],
                  args: &[@ast::expr]) -> @ast::expr {
    fn make_new_lit(cx: &ext_ctxt, sp: span, lit: ast::lit_) -> @ast::expr {
        let sp_lit = @{node: lit, span: sp};
        ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
    }
    fn make_new_str(cx: &ext_ctxt, sp: span, s: str) -> @ast::expr {
        let lit = ast::lit_str(s, ast::sk_rc);
        ret make_new_lit(cx, sp, lit);
    }
    fn make_new_int(cx: &ext_ctxt, sp: span, i: int) -> @ast::expr {
        let lit = ast::lit_int(i);
        ret make_new_lit(cx, sp, lit);
    }
    fn make_new_uint(cx: &ext_ctxt, sp: span, u: uint) -> @ast::expr {
        let lit = ast::lit_uint(u);
        ret make_new_lit(cx, sp, lit);
    }
    fn make_add_expr(cx: &ext_ctxt, sp: span, lhs: @ast::expr,
                     rhs: @ast::expr) -> @ast::expr {
        let binexpr = ast::expr_binary(ast::add, lhs, rhs);
        ret @{id: cx.next_id(), node: binexpr, span: sp};
    }
    fn make_path_expr(cx: &ext_ctxt, sp: span, idents: &[ast::ident]) ->
       @ast::expr {
        let path = {global: false, idents: idents, types: ~[]};
        let sp_path = {node: path, span: sp};
        let pathexpr = ast::expr_path(sp_path);
        ret @{id: cx.next_id(), node: pathexpr, span: sp};
    }
    fn make_vec_expr(cx: &ext_ctxt, sp: span, exprs: &[@ast::expr]) ->
       @ast::expr {
        let vecexpr = ast::expr_vec(exprs, ast::imm);
        ret @{id: cx.next_id(), node: vecexpr, span: sp};
    }
    fn make_call(cx: &ext_ctxt, sp: span, fn_path: &[ast::ident],
                 args: &[@ast::expr]) -> @ast::expr {
        let pathexpr = make_path_expr(cx, sp, fn_path);
        let callexpr = ast::expr_call(pathexpr, args);
        ret @{id: cx.next_id(), node: callexpr, span: sp};
    }
    fn make_rec_expr(cx: &ext_ctxt, sp: span,
                     fields: &[{ident: ast::ident, ex: @ast::expr}]) ->
       @ast::expr {
        let astfields: [ast::field] = ~[];
        for field: {ident: ast::ident, ex: @ast::expr} in fields {
            let ident = field.ident;
            let val = field.ex;
            let astfield =
                {node: {mut: ast::imm, ident: ident, expr: val}, span: sp};
            astfields += ~[astfield];
        }
        let recexpr = ast::expr_rec(astfields, option::none::<@ast::expr>);
        ret @{id: cx.next_id(), node: recexpr, span: sp};
    }
    fn make_path_vec(cx: &ext_ctxt, ident: str) -> [str] {
        fn compiling_std(cx: &ext_ctxt) -> bool {
            ret str::find(cx.crate_file_name(), "std.rc") >= 0;
        }
        if compiling_std(cx) {
            ret ~["extfmt", "rt", ident];
        } else { ret ~["std", "extfmt", "rt", ident]; }
    }
    fn make_rt_path_expr(cx: &ext_ctxt, sp: span, ident: str) -> @ast::expr {
        let path = make_path_vec(cx, ident);
        ret make_path_expr(cx, sp, path);
    }
    // Produces an AST expression that represents a RT::conv record,
    // which tells the RT::conv* functions how to perform the conversion

    fn make_rt_conv_expr(cx: &ext_ctxt, sp: span, cnv: &conv) -> @ast::expr {
        fn make_flags(cx: &ext_ctxt, sp: span, flags: &[flag]) ->
           @ast::expr {
            let flagexprs: [@ast::expr] = ~[];
            for f: flag in flags {
                let fstr;
                alt f {
                  flag_left_justify. { fstr = "flag_left_justify"; }
                  flag_left_zero_pad. { fstr = "flag_left_zero_pad"; }
                  flag_space_for_sign. { fstr = "flag_space_for_sign"; }
                  flag_sign_always. { fstr = "flag_sign_always"; }
                  flag_alternate. { fstr = "flag_alternate"; }
                }
                flagexprs += ~[make_rt_path_expr(cx, sp, fstr)];
            }
            // FIXME: 0-length vectors can't have their type inferred
            // through the rec that these flags are a member of, so
            // this is a hack placeholder flag

            if vec::len::<@ast::expr>(flagexprs) == 0u {
                flagexprs += ~[make_rt_path_expr(cx, sp, "flag_none")];
            }
            ret make_vec_expr(cx, sp, flagexprs);
        }
        fn make_count(cx: &ext_ctxt, sp: span, cnt: &count) -> @ast::expr {
            alt cnt {
              count_implied. {
                ret make_rt_path_expr(cx, sp, "count_implied");
              }
              count_is(c) {
                let count_lit = make_new_int(cx, sp, c);
                let count_is_path = make_path_vec(cx, "count_is");
                let count_is_args = ~[count_lit];
                ret make_call(cx, sp, count_is_path, count_is_args);
              }
              _ { cx.span_unimpl(sp, "unimplemented #fmt conversion"); }
            }
        }
        fn make_ty(cx: &ext_ctxt, sp: span, t: &ty) -> @ast::expr {
            let rt_type;
            alt t {
              ty_hex(c) {
                alt c {
                  case_upper. { rt_type = "ty_hex_upper"; }
                  case_lower. { rt_type = "ty_hex_lower"; }
                }
              }
              ty_bits. { rt_type = "ty_bits"; }
              ty_octal. { rt_type = "ty_octal"; }
              _ { rt_type = "ty_default"; }
            }
            ret make_rt_path_expr(cx, sp, rt_type);
        }
        fn make_conv_rec(cx: &ext_ctxt, sp: span, flags_expr: @ast::expr,
                         width_expr: @ast::expr, precision_expr: @ast::expr,
                         ty_expr: @ast::expr) -> @ast::expr {
            ret make_rec_expr(cx, sp,
                              ~[{ident: "flags", ex: flags_expr},
                                {ident: "width", ex: width_expr},
                                {ident: "precision", ex: precision_expr},
                                {ident: "ty", ex: ty_expr}]);
        }
        let rt_conv_flags = make_flags(cx, sp, cnv.flags);
        let rt_conv_width = make_count(cx, sp, cnv.width);
        let rt_conv_precision = make_count(cx, sp, cnv.precision);
        let rt_conv_ty = make_ty(cx, sp, cnv.ty);
        ret make_conv_rec(cx, sp, rt_conv_flags, rt_conv_width,
                          rt_conv_precision, rt_conv_ty);
    }
    fn make_conv_call(cx: &ext_ctxt, sp: span, conv_type: str, cnv: &conv,
                      arg: @ast::expr) -> @ast::expr {
        let fname = "conv_" + conv_type;
        let path = make_path_vec(cx, fname);
        let cnv_expr = make_rt_conv_expr(cx, sp, cnv);
        let args = ~[cnv_expr, arg];
        ret make_call(cx, arg.span, path, args);
    }
    fn make_new_conv(cx: &ext_ctxt, sp: span, cnv: conv, arg: @ast::expr) ->
       @ast::expr {
        // FIXME: Extract all this validation into extfmt::ct

        fn is_signed_type(cnv: conv) -> bool {
            alt cnv.ty {
              ty_int(s) {
                alt s { signed. { ret true; } unsigned. { ret false; } }
              }
              _ { ret false; }
            }
        }
        let unsupported = "conversion not supported in #fmt string";
        alt cnv.param {
          option::none. { }
          _ { cx.span_unimpl(sp, unsupported); }
        }
        for f: flag in cnv.flags {
            alt f {
              flag_left_justify. { }
              flag_sign_always. {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  "+ flag only valid in " +
                                      "signed #fmt conversion");
                }
              }
              flag_space_for_sign. {
                if !is_signed_type(cnv) {
                    cx.span_fatal(sp,
                                  "space flag only valid in " +
                                      "signed #fmt conversions");
                }
              }
              flag_left_zero_pad. { }
              _ { cx.span_unimpl(sp, unsupported); }
            }
        }
        alt cnv.width {
          count_implied. { }
          count_is(_) { }
          _ { cx.span_unimpl(sp, unsupported); }
        }
        alt cnv.precision {
          count_implied. { }
          count_is(_) { }
          _ { cx.span_unimpl(sp, unsupported); }
        }
        alt cnv.ty {
          ty_str. { ret make_conv_call(cx, arg.span, "str", cnv, arg); }
          ty_int(sign) {
            alt sign {
              signed. { ret make_conv_call(cx, arg.span, "int", cnv, arg); }
              unsigned. {
                ret make_conv_call(cx, arg.span, "uint", cnv, arg);
              }
            }
          }
          ty_bool. { ret make_conv_call(cx, arg.span, "bool", cnv, arg); }
          ty_char. { ret make_conv_call(cx, arg.span, "char", cnv, arg); }
          ty_hex(_) { ret make_conv_call(cx, arg.span, "uint", cnv, arg); }
          ty_bits. { ret make_conv_call(cx, arg.span, "uint", cnv, arg); }
          ty_octal. { ret make_conv_call(cx, arg.span, "uint", cnv, arg); }
          _ { cx.span_unimpl(sp, unsupported); }
        }
    }
    fn log_conv(c: conv) {
        alt c.param {
          some(p) { log "param: " + std::int::to_str(p, 10u); }
          _ { log "param: none"; }
        }
        for f: flag in c.flags {
            alt f {
              flag_left_justify. { log "flag: left justify"; }
              flag_left_zero_pad. { log "flag: left zero pad"; }
              flag_space_for_sign. { log "flag: left space pad"; }
              flag_sign_always. { log "flag: sign always"; }
              flag_alternate. { log "flag: alternate"; }
            }
        }
        alt c.width {
          count_is(i) { log "width: count is " + std::int::to_str(i, 10u); }
          count_is_param(i) {
            log "width: count is param " + std::int::to_str(i, 10u);
          }
          count_is_next_param. { log "width: count is next param"; }
          count_implied. { log "width: count is implied"; }
        }
        alt c.precision {
          count_is(i) { log "prec: count is " + std::int::to_str(i, 10u); }
          count_is_param(i) {
            log "prec: count is param " + std::int::to_str(i, 10u);
          }
          count_is_next_param. { log "prec: count is next param"; }
          count_implied. { log "prec: count is implied"; }
        }
        alt c.ty {
          ty_bool. { log "type: bool"; }
          ty_str. { log "type: str"; }
          ty_char. { log "type: char"; }
          ty_int(s) {
            alt s {
              signed. { log "type: signed"; }
              unsigned. { log "type: unsigned"; }
            }
          }
          ty_bits. { log "type: bits"; }
          ty_hex(cs) {
            alt cs {
              case_upper. { log "type: uhex"; }
              case_lower. { log "type: lhex"; }
            }
          }
          ty_octal. { log "type: octal"; }
        }
    }
    let fmt_sp = args.(0).span;
    let n = 0u;
    let tmp_expr = make_new_str(cx, sp, "");
    let nargs = vec::len::<@ast::expr>(args);
    for pc: piece in pieces {
        alt pc {
          piece_string(s) {
            let s_expr = make_new_str(cx, fmt_sp, s);
            tmp_expr = make_add_expr(cx, fmt_sp, tmp_expr, s_expr);
          }
          piece_conv(conv) {
            n += 1u;
            if n >= nargs {
                cx.span_fatal(sp,
                              "not enough arguments to #fmt " +
                                  "for the given format string");
            }
            log "Building conversion:";
            log_conv(conv);
            let arg_expr = args.(n);
            let c_expr = make_new_conv(cx, fmt_sp, conv, arg_expr);
            tmp_expr = make_add_expr(cx, fmt_sp, tmp_expr, c_expr);
          }
        }
    }
    let expected_nargs = n + 1u; // n conversions + the fmt string

    if expected_nargs < nargs {
        cx.span_fatal
            (sp, #fmt("too many arguments to #fmt. found %u, expected %u",
                      nargs, expected_nargs));
    }
    ret tmp_expr;
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
