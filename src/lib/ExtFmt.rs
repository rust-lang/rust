import option.none;
import option.some;

// Functions used by the fmt extension at compile time
mod CT {
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

    fn parse_fmt_string(str s) -> vec[piece] {
        let vec[piece] pieces = vec();
        auto lim = _str.byte_len(s);
        auto buf = "";

        fn flush_buf(str buf, &vec[piece] pieces) -> str {
            if (_str.byte_len(buf) > 0u) {
                auto piece = piece_string(buf);
                pieces += vec(piece);
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
                    pieces += vec(res._0);
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

    fn peek_num(str s, uint i, uint lim) -> option.t[tup(uint, uint)] {
        if (i >= lim) {
            ret none[tup(uint, uint)];
        }

        auto c = s.(i);
        if (!('0' as u8 <= c && c <= '9' as u8)) {
            ret option.none[tup(uint, uint)];
        }

        auto n = (c - ('0' as u8)) as uint;
        alt (peek_num(s, i + 1u, lim)) {
            case (none[tup(uint, uint)]) {
                ret some[tup(uint, uint)](tup(n, i + 1u));
            }
            case (some[tup(uint, uint)](?next)) {
                auto m = next._0;
                auto j = next._1;
                ret some[tup(uint, uint)](tup(n * 10u + m, j));
            }
        }

    }

    fn parse_conversion(str s, uint i, uint lim) -> tup(piece, uint) {
        auto parm = parse_parameter(s, i, lim);
        auto flags = parse_flags(s, parm._1, lim);
        auto width = parse_count(s, flags._1, lim);
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
            case (none[tup(uint, uint)]) {
                ret tup(none[int], i);
            }
            case (some[tup(uint, uint)](?t)) {
                auto n = t._0;
                auto j = t._1;
                if (j < lim && s.(j) == '$' as u8) {
                    ret tup(some[int](n as int), j + 1u);
                }
                else {
                    ret tup(none[int], i);
                }
            }
        }
    }

    fn parse_flags(str s, uint i, uint lim) -> tup(vec[flag], uint) {
        let vec[flag] noflags = vec();

        if (i >= lim) {
            ret tup(noflags, i);
        }

        fn more_(flag f, str s, uint i, uint lim) -> tup(vec[flag], uint) {
            auto next = parse_flags(s, i + 1u, lim);
            auto rest = next._0;
            auto j = next._1;
            let vec[flag] curr = vec(f);
            ret tup(curr + rest, j);
        }

        auto more = bind more_(_, s, i, lim);

        auto f = s.(i);
        if (f == ('-' as u8)) {
            ret more(flag_left_justify);
        } else if (f == ('0' as u8)) {
            ret more(flag_left_zero_pad);
        } else if (f == (' ' as u8)) {
            ret more(flag_left_space_pad);
        } else if (f == ('+' as u8)) {
            ret more(flag_plus_if_positive);
        } else if (f == ('#' as u8)) {
            ret more(flag_alternate);
        } else {
            ret tup(noflags, i);
        }
    }

    fn parse_count(str s, uint i, uint lim) -> tup(count, uint) {
        if (i >= lim) {
            ret tup(count_implied, i);
        }

        if (s.(i) == ('*' as u8)) {
            auto param = parse_parameter(s, i + 1u, lim);
            auto j = param._1;
            alt (param._0) {
                case (none[int]) {
                    ret tup(count_is_next_param, j);
                }
                case (some[int](?n)) {
                    ret tup(count_is_param(n), j);
                }
            }
        } else {
            auto num = peek_num(s, i, lim);
            alt (num) {
                case (none[tup(uint, uint)]) {
                    ret tup(count_implied, i);
                }
                case (some[tup(uint, uint)](?num)) {
                    ret tup(count_is(num._0 as int), num._1);
                }
            }
        }
    }

    fn parse_precision(str s, uint i, uint lim) -> tup(count, uint) {
        if (i >= lim) {
            ret tup(count_implied, i);
        }

        if (s.(i) == '.' as u8) {
            ret parse_count(s, i + 1u, lim);
        } else {
            ret tup(count_implied, i);
        }
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
            log "unknown type in conversion";
            fail;
        }

        ret tup(t, i + 1u);
    }
}

// Functions used by the fmt extension at runtime
mod RT {
    fn conv_int(int i) -> str {
        ret _int.to_str(i, 10u);
    }

    fn conv_uint(uint u) -> str {
        ret _uint.to_str(u, 10u);
    }

    fn conv_bool(bool b) -> str {
        if (b) {
            ret "true";
        } else {
            ret "false";
        }
    }

    fn conv_char(char c) -> str {
        ret _str.from_char(c);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
