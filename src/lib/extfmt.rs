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

import option::none;
import option::some;

/*
 * We have a CT (compile-time) module that parses format strings into a
 * sequence of conversions. From those conversions AST fragments are built
 * that call into properly-typed functions in the RT (run-time) module.  Each
 * of those run-time conversion functions accepts another conversion
 * description that specifies how to format its output.
 *
 * The building of the AST is currently done in a module inside the compiler,
 * but should migrate over here as the plugin interface is defined.
 */

// Functions used by the fmt extension at compile time
mod ct {
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
        ty_octal;
        // FIXME: More types
    }

    tag flag {
        flag_left_justify;
        flag_left_zero_pad;
        flag_space_for_sign;
        flag_sign_always;
        flag_alternate;
    }

    tag count {
        count_is(int);
        count_is_param(int);
        count_is_next_param;
        count_implied;
    }

    // A formatted conversion from an expression to a string
    type conv = rec(option::t[int] param,
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
        auto lim = _str::byte_len(s);
        auto buf = "";

        fn flush_buf(str buf, &vec[piece] pieces) -> str {
            if (_str::byte_len(buf) > 0u) {
                auto piece = piece_string(buf);
                pieces += vec(piece);
            }
            ret "";
        }

        auto i = 0u;
        while (i < lim) {
            auto curr = _str::substr(s, i, 1u);
            if (_str::eq(curr, "%")) {
                i += 1u;
                if (i >= lim) {
                    log_err "unterminated conversion at end of string";
                    fail;
                }
                auto curr2 = _str::substr(s, i, 1u);
                if (_str::eq(curr2, "%")) {
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

    fn peek_num(str s, uint i, uint lim) -> option::t[tup(uint, uint)] {
        if (i >= lim) {
            ret none[tup(uint, uint)];
        }

        auto c = s.(i);
        if (!('0' as u8 <= c && c <= '9' as u8)) {
            ret option::none[tup(uint, uint)];
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

    fn parse_parameter(str s, uint i, uint lim) -> tup(option::t[int], uint) {
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
            ret more(flag_space_for_sign);
        } else if (f == ('+' as u8)) {
            ret more(flag_sign_always);
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
            auto count = parse_count(s, i + 1u, lim);
            // If there were no digits specified, i.e. the precision
            // was ".", then the precision is 0
            alt (count._0) {
                case (count_implied) {
                    ret tup(count_is(0), count._1);
                }
                case (_) {
                    ret count;
                }
            }
        } else {
            ret tup(count_implied, i);
        }
    }

    fn parse_type(str s, uint i, uint lim) -> tup(ty, uint) {
        if (i >= lim) {
            log_err "missing type in conversion";
            fail;
        }

        auto t;
        auto tstr = _str::substr(s, i, 1u);
        if (_str::eq(tstr, "b")) {
            t = ty_bool;
        } else if (_str::eq(tstr, "s")) {
            t = ty_str;
        } else if (_str::eq(tstr, "c")) {
            t = ty_char;
        } else if (_str::eq(tstr, "d")
                   || _str::eq(tstr, "i")) {
            // TODO: Do we really want two signed types here?
            // How important is it to be printf compatible?
            t = ty_int(signed);
        } else if (_str::eq(tstr, "u")) {
            t = ty_int(unsigned);
        } else if (_str::eq(tstr, "x")) {
            t = ty_hex(case_lower);
        } else if (_str::eq(tstr, "X")) {
            t = ty_hex(case_upper);
        } else if (_str::eq(tstr, "t")) {
            t = ty_bits;
        } else if (_str::eq(tstr, "o")) {
            t = ty_octal;
        } else {
            log_err "unknown type in conversion";
            fail;
        }

        ret tup(t, i + 1u);
    }
}

// Functions used by the fmt extension at runtime. For now there are a lot of
// decisions made a runtime. If it proves worthwhile then some of these
// conditions can be evaluated at compile-time. For now though it's cleaner to
// implement it this way, I think.
mod rt {

    tag flag {
        flag_left_justify;
        flag_left_zero_pad;
        flag_space_for_sign;
        flag_sign_always;
        flag_alternate;
        // FIXME: This is a hack to avoid creating 0-length vec exprs,
        // which have some difficulty typechecking currently. See
        // comments in front::extfmt::make_flags
        flag_none;
    }

    tag count {
        count_is(int);
        count_implied;
    }

    tag ty {
        ty_default;
        ty_bits;
        ty_hex_upper;
        ty_hex_lower;
        ty_octal;
    }

    // FIXME: May not want to use a vector here for flags;
    // instead just use a bool per flag
    type conv = rec(vec[flag] flags,
                    count width,
                    count precision,
                    ty ty);

    fn conv_int(&conv cv, int i) -> str {
        auto radix = 10u;
        auto prec = get_int_precision(cv);
        auto s = int_to_str_prec(i, radix, prec);
        if (0 <= i) {
            if (have_flag(cv.flags, flag_sign_always)) {
                s = "+" + s;
            } else if (have_flag(cv.flags, flag_space_for_sign)) {
                s = " " + s;
            }
        }
        ret pad(cv, s, pad_signed);
    }

    fn conv_uint(&conv cv, uint u) -> str {
        auto prec = get_int_precision(cv);
        auto res;
        alt (cv.ty) {
            case (ty_default) {
                res = uint_to_str_prec(u, 10u, prec);
            }
            case (ty_hex_lower) {
                res = uint_to_str_prec(u, 16u, prec);
            }
            case (ty_hex_upper) {
                res = _str::to_upper(uint_to_str_prec(u, 16u, prec));
            }
            case (ty_bits) {
                res = uint_to_str_prec(u, 2u, prec);
            }
            case (ty_octal) {
                res = uint_to_str_prec(u, 8u, prec);
            }
        }
        ret pad(cv, res, pad_unsigned);
    }

    fn conv_bool(&conv cv, bool b) -> str {
        auto s;
        if (b) {
            s = "true";
        } else {
            s = "false";
        }
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        ret conv_str(cv, s);
    }

    fn conv_char(&conv cv, char c) -> str {
        ret pad(cv, _str::from_char(c), pad_nozero);
    }

    fn conv_str(&conv cv, str s) -> str {
        auto unpadded = s;
        alt (cv.precision) {
            case (count_implied) {
            }
            case (count_is(?max)) {
                // For strings, precision is the maximum characters displayed
                if (max as uint < _str::char_len(s)) {
                    // FIXME: substr works on bytes, not chars!
                    unpadded = _str::substr(s, 0u, max as uint);
                }
            }
        }
        ret pad(cv, unpadded, pad_nozero);
    }

    // Convert an int to string with minimum number of digits. If precision is
    // 0 and num is 0 then the result is the empty string.
    fn int_to_str_prec(int num, uint radix, uint prec) -> str {
        if (num < 0) {
            ret "-" + uint_to_str_prec((-num) as uint, radix, prec);
        } else {
            ret uint_to_str_prec(num as uint, radix, prec);
        }
    }

    // Convert a uint to string with a minimum number of digits.  If precision
    // is 0 and num is 0 then the result is the empty string. Could move this
    // to _uint: but it doesn't seem all that useful.
    fn uint_to_str_prec(uint num, uint radix, uint prec) -> str {
        auto s;

        if (prec == 0u && num == 0u) {
            s = "";
        } else {
            s = _uint::to_str(num, radix);
            auto len = _str::char_len(s);
            if (len < prec) {
                auto diff = prec - len;
                auto pad = str_init_elt('0', diff);
                s = pad + s;
            }
        }

        ret s;
    }

    fn get_int_precision(&conv cv) -> uint {
        alt (cv.precision) {
            case (count_is(?c)) {
                ret c as uint;
            }
            case (count_implied) {
                ret 1u;
            }
        }
    }

    // FIXME: This might be useful in _str: but needs to be utf8 safe first
    fn str_init_elt(char c, uint n_elts) -> str {
        auto svec = _vec::init_elt[u8](c as u8, n_elts);
        // FIXME: Using unsafe_from_bytes because rustboot
        // can't figure out the is_utf8 predicate on from_bytes?
        ret _str::unsafe_from_bytes(svec);
    }

    tag pad_mode {
        pad_signed;
        pad_unsigned;
        pad_nozero;
    }

    fn pad(&conv cv, str s, pad_mode mode) -> str {
        auto uwidth;
        alt (cv.width) {
            case (count_implied) {
                ret s;
            }
            case (count_is(?width)) {
                // FIXME: Maybe width should be uint
                uwidth = width as uint;
            }
        }

        auto strlen = _str::char_len(s);
        if (uwidth <= strlen) {
            ret s;
        }

        auto padchar = ' ';
        auto diff = uwidth - strlen;
        if (have_flag(cv.flags, flag_left_justify)) {
            auto padstr = str_init_elt(padchar, diff);
            ret s + padstr;
        }

        auto might_zero_pad = false;
        auto signed = false;

        alt (mode) {
            case (pad_nozero) {
                // fallthrough
            }
            case (pad_signed) {
                might_zero_pad = true;
                signed = true;
            }
            case (pad_unsigned) {
                might_zero_pad = true;
            }
        }

        fn have_precision(&conv cv) -> bool {
            alt (cv.precision) {
                case (count_implied) {
                    ret false;
                }
                case (_) {
                    ret true;
                }
            }
        }

        auto zero_padding = false;
        if (might_zero_pad
            && have_flag(cv.flags, flag_left_zero_pad)
            && !have_precision(cv)) {

            padchar = '0';
            zero_padding = true;
        }

        auto padstr = str_init_elt(padchar, diff);

        // This is completely heinous. If we have a signed value then
        // potentially rip apart the intermediate result and insert some
        // zeros. It may make sense to convert zero padding to a precision
        // instead.
        if (signed
            && zero_padding
            && _str::byte_len(s) > 0u) {

            auto head = s.(0);
            if (head == '+' as u8
                || head == '-' as u8
                || head == ' ' as u8) {

                auto headstr = _str::unsafe_from_bytes(vec(head));
                auto bytelen = _str::byte_len(s);
                auto numpart = _str::substr(s, 1u, bytelen - 1u);
                ret headstr + padstr + numpart;
            }
        }
        ret padstr + s;
    }

    fn have_flag(vec[flag] flags, flag f) -> bool {
        for (flag candidate in flags) {
            if (candidate == f) {
                ret true;
            }
        }
        ret false;
    }
}

// FIXME: This is a temporary duplication of the rt mod that only
// needs to exist to accomodate the stage0 compiler until the next snapshot
mod RT {
    tag flag {
        flag_left_justify;
        flag_left_zero_pad;
        flag_space_for_sign;
        flag_sign_always;
        flag_alternate;
        // FIXME: This is a hack to avoid creating 0-length vec exprs,
        // which have some difficulty typechecking currently. See
        // comments in front::extfmt::make_flags
        flag_none;
    }

    tag count {
        count_is(int);
        count_implied;
    }

    tag ty {
        ty_default;
        ty_bits;
        ty_hex_upper;
        ty_hex_lower;
        ty_octal;
    }

    // FIXME: May not want to use a vector here for flags;
    // instead just use a bool per flag
    type conv = rec(vec[flag] flags,
                    count width,
                    count precision,
                    ty ty);

    fn conv_int(&conv cv, int i) -> str {
        auto radix = 10u;
        auto prec = get_int_precision(cv);
        auto s = int_to_str_prec(i, radix, prec);
        if (0 <= i) {
            if (have_flag(cv.flags, flag_sign_always)) {
                s = "+" + s;
            } else if (have_flag(cv.flags, flag_space_for_sign)) {
                s = " " + s;
            }
        }
        ret pad(cv, s, pad_signed);
    }

    fn conv_uint(&conv cv, uint u) -> str {
        auto prec = get_int_precision(cv);
        auto res;
        alt (cv.ty) {
            case (ty_default) {
                res = uint_to_str_prec(u, 10u, prec);
            }
            case (ty_hex_lower) {
                res = uint_to_str_prec(u, 16u, prec);
            }
            case (ty_hex_upper) {
                res = _str::to_upper(uint_to_str_prec(u, 16u, prec));
            }
            case (ty_bits) {
                res = uint_to_str_prec(u, 2u, prec);
            }
            case (ty_octal) {
                res = uint_to_str_prec(u, 8u, prec);
            }
        }
        ret pad(cv, res, pad_unsigned);
    }

    fn conv_bool(&conv cv, bool b) -> str {
        auto s;
        if (b) {
            s = "true";
        } else {
            s = "false";
        }
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        ret conv_str(cv, s);
    }

    fn conv_char(&conv cv, char c) -> str {
        ret pad(cv, _str::from_char(c), pad_nozero);
    }

    fn conv_str(&conv cv, str s) -> str {
        auto unpadded = s;
        alt (cv.precision) {
            case (count_implied) {
            }
            case (count_is(?max)) {
                // For strings, precision is the maximum characters displayed
                if (max as uint < _str::char_len(s)) {
                    // FIXME: substr works on bytes, not chars!
                    unpadded = _str::substr(s, 0u, max as uint);
                }
            }
        }
        ret pad(cv, unpadded, pad_nozero);
    }

    // Convert an int to string with minimum number of digits. If precision is
    // 0 and num is 0 then the result is the empty string.
    fn int_to_str_prec(int num, uint radix, uint prec) -> str {
        if (num < 0) {
            ret "-" + uint_to_str_prec((-num) as uint, radix, prec);
        } else {
            ret uint_to_str_prec(num as uint, radix, prec);
        }
    }

    // Convert a uint to string with a minimum number of digits.  If precision
    // is 0 and num is 0 then the result is the empty string. Could move this
    // to _uint: but it doesn't seem all that useful.
    fn uint_to_str_prec(uint num, uint radix, uint prec) -> str {
        auto s;

        if (prec == 0u && num == 0u) {
            s = "";
        } else {
            s = _uint::to_str(num, radix);
            auto len = _str::char_len(s);
            if (len < prec) {
                auto diff = prec - len;
                auto pad = str_init_elt('0', diff);
                s = pad + s;
            }
        }

        ret s;
    }

    fn get_int_precision(&conv cv) -> uint {
        alt (cv.precision) {
            case (count_is(?c)) {
                ret c as uint;
            }
            case (count_implied) {
                ret 1u;
            }
        }
    }

    // FIXME: This might be useful in _str: but needs to be utf8 safe first
    fn str_init_elt(char c, uint n_elts) -> str {
        auto svec = _vec::init_elt[u8](c as u8, n_elts);
        // FIXME: Using unsafe_from_bytes because rustboot
        // can't figure out the is_utf8 predicate on from_bytes?
        ret _str::unsafe_from_bytes(svec);
    }

    tag pad_mode {
        pad_signed;
        pad_unsigned;
        pad_nozero;
    }

    fn pad(&conv cv, str s, pad_mode mode) -> str {
        auto uwidth;
        alt (cv.width) {
            case (count_implied) {
                ret s;
            }
            case (count_is(?width)) {
                // FIXME: Maybe width should be uint
                uwidth = width as uint;
            }
        }

        auto strlen = _str::char_len(s);
        if (uwidth <= strlen) {
            ret s;
        }

        auto padchar = ' ';
        auto diff = uwidth - strlen;
        if (have_flag(cv.flags, flag_left_justify)) {
            auto padstr = str_init_elt(padchar, diff);
            ret s + padstr;
        }

        auto might_zero_pad = false;
        auto signed = false;

        alt (mode) {
            case (pad_nozero) {
                // fallthrough
            }
            case (pad_signed) {
                might_zero_pad = true;
                signed = true;
            }
            case (pad_unsigned) {
                might_zero_pad = true;
            }
        }

        fn have_precision(&conv cv) -> bool {
            alt (cv.precision) {
                case (count_implied) {
                    ret false;
                }
                case (_) {
                    ret true;
                }
            }
        }

        auto zero_padding = false;
        if (might_zero_pad
            && have_flag(cv.flags, flag_left_zero_pad)
            && !have_precision(cv)) {

            padchar = '0';
            zero_padding = true;
        }

        auto padstr = str_init_elt(padchar, diff);

        // This is completely heinous. If we have a signed value then
        // potentially rip apart the intermediate result and insert some
        // zeros. It may make sense to convert zero padding to a precision
        // instead.
        if (signed
            && zero_padding
            && _str::byte_len(s) > 0u) {

            auto head = s.(0);
            if (head == '+' as u8
                || head == '-' as u8
                || head == ' ' as u8) {

                auto headstr = _str::unsafe_from_bytes(vec(head));
                auto bytelen = _str::byte_len(s);
                auto numpart = _str::substr(s, 1u, bytelen - 1u);
                ret headstr + padstr + numpart;
            }
        }
        ret padstr + s;
    }

    fn have_flag(vec[flag] flags, flag f) -> bool {
        for (flag candidate in flags) {
            if (candidate == f) {
                ret true;
            }
        }
        ret false;
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
