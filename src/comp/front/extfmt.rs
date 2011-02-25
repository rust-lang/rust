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

import front.parser;

import std._str;
import std._vec;
import std.option;

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
tag conv {
    conv_param(option.t[int]);
    conv_flags(vec[flag]);
    conv_width(count);
    conv_precision(count);
    conv_ty(ty);
}

// A fragment of the output sequence
tag piece {
    piece_string(str);
    piece_conv(conv);
}

fn bad_fmt_call() {
    log "malformed #fmt call";
    fail;
}

fn expand_syntax_ext(vec[@ast.expr] args,
                     option.t[@ast.expr] body) -> @ast.expr {

    if (_vec.len[@ast.expr](args) == 0u) {
        bad_fmt_call();
    }

    auto fmt = expr_to_str(args.(0));
    log fmt;
    auto pieces = parse_fmt_string(fmt);
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

    // TODO: This is super ugly
    fn flush_buf(str buf, vec[piece] pieces) -> str {
        log "flushing";
        if (_str.byte_len(buf) > 0u) {
            auto piece = piece_string(buf);
            pieces += piece;
        }
        log "buf:";
        log buf;
        log "pieces:";
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
        ret "";
    }

    auto i = 0u;
    while (i < lim) {
        log "step:";
        log i;
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
            }
        } else {
            buf += curr;
            log "buf:";
            log buf;
            i += 1u;
        }
    }

    ret pieces;
}

fn pieces_to_expr(vec[piece] pieces, vec[@ast.expr] args) -> @ast.expr {
    auto lo = args.(0).span;
    auto hi = args.(0).span;
    auto strlit = ast.lit_str("TODO");
    auto spstrlit = @parser.spanned[ast.lit_](lo, hi, strlit);
    auto expr = ast.expr_lit(spstrlit, ast.ann_none);
    auto spexpr = @parser.spanned[ast.expr_](lo, hi, expr);
    ret spexpr;
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
