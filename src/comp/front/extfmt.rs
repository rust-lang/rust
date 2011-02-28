/* The 'fmt' extension is modeled on the posix printf system.
 * 
 * A posix conversion ostensibly looks like this:
 * 
 * %[parameter][flags][width][.precision][length]type
 * 
 * Given the different numeric type bestiary we have, we omit the 'length'
 * parameter and support slightly different conversions for 'type':
 * 
 * %[parameter][flags][width][.precision]type
 * 
 * we also only support translating-to-rust a tiny subset of the possible
 * combinations at the moment.
 */

import util.common;

import std._str;
import std._vec;
import std.option;
import std.option.none;
import std.option.some;

export expand_syntax_ext;

tag signedness {
    signed;
    unsigned;
}

tag caseness {
    case_upper;
    case_lower;
}

tag ty {
    ty_bool;
    ty_str;
    ty_char;
    ty_int(signedness);
    ty_bits;
    ty_hex(caseness);
    // FIXME: More types
}

tag flag {
    flag_left_justify;
    flag_left_zero_pad;
    flag_left_space_pad;
    flag_plus_if_positive;
    flag_alternate;
}

tag count {
    count_is(int);
    count_is_param(int);
    count_is_next_param;
    count_implied;
}

// A formatted conversion from an expression to a string
type conv = rec(option.t[int] param,
                vec[flag] flags,
                count width,
                count precision,
                ty ty);

// A fragment of the output sequence
tag piece {
    piece_string(str);
    piece_conv(conv);
}

fn bad_fmt_call() {
    log "malformed #fmt call";
    fail;
}

// TODO: Need to thread parser through here to handle errors correctly
fn expand_syntax_ext(vec[@ast.expr] args,
                     option.t[@ast.expr] body) -> @ast.expr {

    if (_vec.len[@ast.expr](args) == 0u) {
        bad_fmt_call();
    }

    auto fmt = expr_to_str(args.(0));
    log fmt;
    auto pieces = parse_fmt_string(fmt);
    log "printing all pieces";
    for (piece p in pieces) {
        alt (p) {
            case (piece_string(?s)) {
                log s;
            }
            case (piece_conv(_)) {
                log "conv";
            }
        }
    }
    log "done printing all pieces";
    auto args_len = _vec.len[@ast.expr](args);
    auto fmt_args = _vec.slice[@ast.expr](args, 1u, args_len - 1u);
    ret pieces_to_expr(pieces, args);
}

fn expr_to_str(@ast.expr expr) -> str {
    alt (expr.node) {
        case (ast.expr_lit(?l, _)) {
            alt (l.node) {
                case (ast.lit_str(?s)) {
                    ret s;
                }
            }
        }
    }
    bad_fmt_call();
    fail;
}

fn parse_fmt_string(str s) -> vec[piece] {
    let vec[piece] pieces = vec();
    // FIXME: Should be counting codepoints instead of bytes
    auto lim = _str.byte_len(s);
    auto buf = "";

    fn flush_buf(str buf, &vec[piece] pieces) -> str {
        if (_str.byte_len(buf) > 0u) {
            auto piece = piece_string(buf);
            pieces += piece;
        }
        ret "";
    }

    auto i = 0u;
    while (i < lim) {
        auto curr = _str.substr(s, i, 1u);
        if (_str.eq(curr, "%")) {
            i += 1u;
            if (i >= lim) {
                log "unterminated conversion at end of string";
                fail;
            }
            auto curr2 = _str.substr(s, i, 1u);
            if (_str.eq(curr2, "%")) {
                i += 1u;
            } else {
                buf = flush_buf(buf, pieces);
                auto res = parse_conversion(s, i, lim);
                pieces += res._0;
                i = res._1;
            }
        } else {
            buf += curr;
            i += 1u;
        }
    }
    buf = flush_buf(buf, pieces);
    ret pieces;
}

fn peek_num(str s, uint i, uint lim) -> option.t[tup(int, int)] {
    if (i >= lim) {
        ret none[tup(int, int)];
    } else {
        ret none[tup(int, int)];
        /*if ('0' <= c && c <= '9') {
            log c;
            fail;
        } else {
            ret option.none[tup(int, int)];
        }
        */
    }
}

fn parse_conversion(str s, uint i, uint lim) -> tup(piece, uint) {
    auto parm = parse_parameter(s, i, lim);
    auto flags = parse_flags(s, parm._1, lim);
    auto width = parse_width(s, flags._1, lim);
    auto prec = parse_precision(s, width._1, lim);
    auto ty = parse_type(s, prec._1, lim);
    ret tup(piece_conv(rec(param = parm._0,
                           flags = flags._0,
                           width = width._0,
                           precision = prec._0,
                           ty = ty._0)),
            ty._1);
}

fn parse_parameter(str s, uint i, uint lim) -> tup(option.t[int], uint) {
    if (i >= lim) {
        ret tup(none[int], i);
    }

    auto num = peek_num(s, i, lim);
    alt (num) {
        case (none[tup(int, int)]) {
            ret tup(none[int], i);
        }
        case (some[tup(int, int)](?t)) {
            fail;
        }
    }
}

fn parse_flags(str s, uint i, uint lim) -> tup(vec[flag], uint) {
    let vec[flag] flags = vec();
    ret tup(flags, i);
}

fn parse_width(str s, uint i, uint lim) -> tup(count, uint) {
    ret tup(count_implied, i);
}

fn parse_precision(str s, uint i, uint lim) -> tup(count, uint) {
    ret tup(count_implied, i);
}

fn parse_type(str s, uint i, uint lim) -> tup(ty, uint) {
    if (i >= lim) {
        log "missing type in conversion";
        fail;
    }

    auto t;
    auto tstr = _str.substr(s, i, 1u);
    if (_str.eq(tstr, "b")) {
        t = ty_bool;
    } else if (_str.eq(tstr, "s")) {
        t = ty_str;
    } else if (_str.eq(tstr, "c")) {
        t = ty_char;
    } else if (_str.eq(tstr, "d")
               || _str.eq(tstr, "i")) {
        // TODO: Do we really want two signed types here?
        // How important is it to be printf compatible?
        t = ty_int(signed);
    } else if (_str.eq(tstr, "u")) {
        t = ty_int(unsigned);
    } else if (_str.eq(tstr, "x")) {
        t = ty_hex(case_lower);
    } else if (_str.eq(tstr, "X")) {
        t = ty_hex(case_upper);
    } else if (_str.eq(tstr, "t")) {
        t = ty_bits;
    } else {
        // FIXME: This is a hack to avoid 'unsatisfied precondition
        // constraint' on uninitialized variable t below
        t = ty_bool;
        log "unknown type in conversion";
        fail;
    }

    ret tup(t, i + 1u);
}

fn pieces_to_expr(vec[piece] pieces, vec[@ast.expr] args) -> @ast.expr {

    fn make_new_lit(common.span sp, ast.lit_ lit) -> @ast.expr {
        auto sp_lit = @parser.spanned[ast.lit_](sp, sp, lit);
        auto expr = ast.expr_lit(sp_lit, ast.ann_none);
        ret @parser.spanned[ast.expr_](sp, sp, expr);
    }

    fn make_new_str(common.span sp, str s) -> @ast.expr {
        auto lit = ast.lit_str(s);
        ret make_new_lit(sp, lit);
    }

    fn make_new_uint(common.span sp, uint u) -> @ast.expr {
        auto lit = ast.lit_uint(u);
        ret make_new_lit(sp, lit);
    }

    fn make_add_expr(common.span sp,
                     @ast.expr lhs, @ast.expr rhs) -> @ast.expr {
        auto binexpr = ast.expr_binary(ast.add, lhs, rhs, ast.ann_none);
        ret @parser.spanned[ast.expr_](sp, sp, binexpr);
    }

    fn make_call(common.span sp, vec[ast.ident] fn_path,
                 vec[@ast.expr] args) -> @ast.expr {
        let vec[ast.ident] path_idents = fn_path;
        let vec[@ast.ty] path_types = vec();
        auto path = rec(idents = path_idents, types = path_types);
        auto sp_path = parser.spanned[ast.path_](sp, sp, path);
        auto pathexpr = ast.expr_path(sp_path, none[ast.def], ast.ann_none);
        auto sp_pathexpr = @parser.spanned[ast.expr_](sp, sp, pathexpr);
        auto callexpr = ast.expr_call(sp_pathexpr, args, ast.ann_none);
        auto sp_callexpr = @parser.spanned[ast.expr_](sp, sp, callexpr);
        ret sp_callexpr;
    }

    fn make_new_conv(conv cnv, @ast.expr arg) -> @ast.expr {

        auto unsupported = "conversion not supported in #fmt string";

        alt (cnv.param) {
            case (option.none[int]) {
            }
            case (_) {
                log unsupported;
                fail;
            }
        }

        if (_vec.len[flag](cnv.flags) != 0u) {
            log unsupported;
            fail;
        }

        alt (cnv.width) {
            case (count_implied) {
            }
            case (_) {
                log unsupported;
                fail;
            }
        }

        alt (cnv.precision) {
            case (count_implied) {
            }
            case (_) {
                log unsupported;
                fail;
            }
        }

        alt (cnv.ty) {
            case (ty_str) {
                ret arg;
            }
            case (ty_int(?sign)) {
                alt (sign) {
                    case (signed) {
                        let vec[str] path = vec("std", "_int", "to_str");
                        auto radix_expr = make_new_uint(arg.span, 10u);
                        let vec[@ast.expr] args = vec(arg, radix_expr);
                        ret make_call(arg.span, path, args);
                    }
                    case (unsigned) {
                        let vec[str] path = vec("std", "_uint", "to_str");
                        auto radix_expr = make_new_uint(arg.span, 10u);
                        let vec[@ast.expr] args = vec(arg, radix_expr);
                        ret make_call(arg.span, path, args);
                    }
                }
            }
            case (_) {
                log unsupported;
                fail;
            }
        }
    }

    auto sp = args.(0).span;
    auto n = 0u;
    auto tmp_expr = make_new_str(sp, "");

    for (piece p in pieces) {
        alt (p) {
            case (piece_string(?s)) {
                auto s_expr = make_new_str(sp, s);
                tmp_expr = make_add_expr(sp, tmp_expr, s_expr);
            }
            case (piece_conv(?conv)) {
                if (n >= _vec.len[@ast.expr](args)) {
                    log "too many conversions in #fmt string";
                    fail;
                }

                n += 1u;
                auto arg_expr = args.(n);
                auto c_expr = make_new_conv(conv, arg_expr);
                tmp_expr = make_add_expr(sp, tmp_expr, c_expr);
            }
        }
    }

    // TODO: Remove this print and return the real expanded AST
    log "dumping expanded ast:";
    log pretty.print_expr(tmp_expr);
    ret make_new_str(sp, "TODO");
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
