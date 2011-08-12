

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
 * We have a 'ct' (compile-time) module that parses format strings into a
 * sequence of conversions. From those conversions AST fragments are built
 * that call into properly-typed functions in the 'rt' (run-time) module.
 * Each of those run-time conversion functions accepts another conversion
 * description that specifies how to format its output.
 *
 * The building of the AST is currently done in a module inside the compiler,
 * but should migrate over here as the plugin interface is defined.
 */

// Functions used by the fmt extension at compile time
mod ct {
    tag signedness { signed; unsigned; }
    tag caseness { case_upper; case_lower; }
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
    type conv =
        {param: option::t<int>,
         flags: [flag],
         width: count,
         precision: count,
         ty: ty};


    // A fragment of the output sequence
    tag piece { piece_string(str); piece_conv(conv); }
    type error_fn = fn(str) -> !  ;

    fn parse_fmt_string(s: str, error: error_fn) -> [piece] {
        let pieces: [piece] = ~[];
        let lim = str::byte_len(s);
        let buf = "";
        fn flush_buf(buf: str, pieces: &mutable [piece]) -> str {
            if str::byte_len(buf) > 0u {
                let piece = piece_string(buf);
                pieces += ~[piece];
            }
            ret "";
        }
        let i = 0u;
        while i < lim {
            let curr = str::substr(s, i, 1u);
            if str::eq(curr, "%") {
                i += 1u;
                if i >= lim {
                    error("unterminated conversion at end of string");
                }
                let curr2 = str::substr(s, i, 1u);
                if str::eq(curr2, "%") {
                    i += 1u;
                } else {
                    buf = flush_buf(buf, pieces);
                    let rs = parse_conversion(s, i, lim, error);
                    pieces += ~[rs.piece];
                    i = rs.next;
                }
            } else { buf += curr; i += 1u; }
        }
        buf = flush_buf(buf, pieces);
        ret pieces;
    }
    fn peek_num(s: str, i: uint, lim: uint) ->
       option::t<{num: uint, next: uint}> {
        if i >= lim { ret none; }
        let c = s.(i);
        if !('0' as u8 <= c && c <= '9' as u8) { ret option::none; }
        let n = c - ('0' as u8) as uint;
        ret alt peek_num(s, i + 1u, lim) {
              none. { some({num: n, next: i + 1u}) }
              some(next) {
                let m = next.num;
                let j = next.next;
                some({num: n * 10u + m, next: j})
              }
            };
    }
    fn parse_conversion(s: str, i: uint, lim: uint, error: error_fn) ->
       {piece: piece, next: uint} {
        let parm = parse_parameter(s, i, lim);
        let flags = parse_flags(s, parm.next, lim);
        let width = parse_count(s, flags.next, lim);
        let prec = parse_precision(s, width.next, lim);
        let ty = parse_type(s, prec.next, lim, error);
        ret {piece:
                 piece_conv({param: parm.param,
                             flags: flags.flags,
                             width: width.count,
                             precision: prec.count,
                             ty: ty.ty}),
             next: ty.next};
    }
    fn parse_parameter(s: str, i: uint, lim: uint) ->
       {param: option::t<int>, next: uint} {
        if i >= lim { ret {param: none, next: i}; }
        let num = peek_num(s, i, lim);
        ret alt num {
              none. { {param: none, next: i} }
              some(t) {
                let n = t.num;
                let j = t.next;
                if j < lim && s.(j) == '$' as u8 {
                    {param: some(n as int), next: j + 1u}
                } else { {param: none, next: i} }
              }
            };
    }
    fn parse_flags(s: str, i: uint, lim: uint) ->
       {flags: [flag], next: uint} {
        let noflags: [flag] = ~[];
        if i >= lim { ret {flags: noflags, next: i}; }

        // FIXME: This recursion generates illegal instructions if the return
        // value isn't boxed. Only started happening after the ivec conversion
        fn more_(f: flag, s: str, i: uint, lim: uint) ->
           @{flags: [flag], next: uint} {
            let next = parse_flags(s, i + 1u, lim);
            let rest = next.flags;
            let j = next.next;
            let curr: [flag] = ~[f];
            ret @{flags: curr + rest, next: j};
        }
        let more = bind more_(_, s, i, lim);
        let f = s.(i);
        ret if f == '-' as u8 {
                *more(flag_left_justify)
            } else if (f == '0' as u8) {
                *more(flag_left_zero_pad)
            } else if (f == ' ' as u8) {
                *more(flag_space_for_sign)
            } else if (f == '+' as u8) {
                *more(flag_sign_always)
            } else if (f == '#' as u8) {
                *more(flag_alternate)
            } else { {flags: noflags, next: i} };
    }
    fn parse_count(s: str, i: uint, lim: uint) -> {count: count, next: uint} {
        ret if i >= lim {
                {count: count_implied, next: i}
            } else if (s.(i) == '*' as u8) {
                let param = parse_parameter(s, i + 1u, lim);
                let j = param.next;
                alt param.param {
                  none. { {count: count_is_next_param, next: j} }
                  some(n) { {count: count_is_param(n), next: j} }
                }
            } else {
                let num = peek_num(s, i, lim);
                alt num {
                  none. { {count: count_implied, next: i} }
                  some(num) {
                    {count: count_is(num.num as int), next: num.next}
                  }
                }
            };
    }
    fn parse_precision(s: str, i: uint, lim: uint) ->
       {count: count, next: uint} {
        ret if i >= lim {
                {count: count_implied, next: i}
            } else if (s.(i) == '.' as u8) {
                let count = parse_count(s, i + 1u, lim);


                // If there were no digits specified, i.e. the precision
                // was ".", then the precision is 0
                alt count.count {
                  count_implied. { {count: count_is(0), next: count.next} }
                  _ { count }
                }
            } else { {count: count_implied, next: i} };
    }
    fn parse_type(s: str, i: uint, lim: uint, error: error_fn) ->
       {ty: ty, next: uint} {
        if i >= lim { error("missing type in conversion"); }
        let tstr = str::substr(s, i, 1u);
        // TODO: Do we really want two signed types here?
        // How important is it to be printf compatible?
        let t =
            if str::eq(tstr, "b") {
                ty_bool
            } else if (str::eq(tstr, "s")) {
                ty_str
            } else if (str::eq(tstr, "c")) {
                ty_char
            } else if (str::eq(tstr, "d") || str::eq(tstr, "i")) {
                ty_int(signed)
            } else if (str::eq(tstr, "u")) {
                ty_int(unsigned)
            } else if (str::eq(tstr, "x")) {
                ty_hex(case_lower)
            } else if (str::eq(tstr, "X")) {
                ty_hex(case_upper)
            } else if (str::eq(tstr, "t")) {
                ty_bits
            } else if (str::eq(tstr, "o")) {
                ty_octal
            } else { error("unknown type in conversion: " + tstr) };
        ret {ty: t, next: i + 1u};
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
    tag count { count_is(int); count_implied; }
    tag ty { ty_default; ty_bits; ty_hex_upper; ty_hex_lower; ty_octal; }

    // FIXME: May not want to use a vector here for flags;
    // instead just use a bool per flag
    type conv = {flags: [flag], width: count, precision: count, ty: ty};

    // FIXME: Remove these transitional *_ivec interfaces
    fn conv_int_ivec(cv: &conv, i: int) -> str {
        conv_int(cv, i)
    }
    fn conv_uint_ivec(cv: &conv, u: uint) -> str {
        conv_uint(cv, u)
    }
    fn conv_bool_ivec(cv: &conv, b: bool) -> str {
        conv_bool(cv, b)
    }
    fn conv_char_ivec(cv: &conv, c: char) -> str {
        conv_char(cv, c)
    }
    fn conv_str_ivec(cv: &conv, s: str) -> str {
        conv_str(cv, s)
    }

    fn conv_int(cv: &conv, i: int) -> str {
        let radix = 10u;
        let prec = get_int_precision(cv);
        let s = int_to_str_prec(i, radix, prec);
        if 0 <= i {
            if have_flag(cv.flags, flag_sign_always) {
                s = "+" + s;
            } else if (have_flag(cv.flags, flag_space_for_sign)) {
                s = " " + s;
            }
        }
        ret pad(cv, s, pad_signed);
    }
    fn conv_uint(cv: &conv, u: uint) -> str {
        let prec = get_int_precision(cv);
        let rs =
            alt cv.ty {
              ty_default. { uint_to_str_prec(u, 10u, prec) }
              ty_hex_lower. { uint_to_str_prec(u, 16u, prec) }
              ty_hex_upper. { str::to_upper(uint_to_str_prec(u, 16u, prec)) }
              ty_bits. { uint_to_str_prec(u, 2u, prec) }
              ty_octal. { uint_to_str_prec(u, 8u, prec) }
            };
        ret pad(cv, rs, pad_unsigned);
    }
    fn conv_bool(cv: &conv, b: bool) -> str {
        let s = if b { "true" } else { "false" };
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.

        ret conv_str_ivec(cv, s);
    }
    fn conv_char(cv: &conv, c: char) -> str {
        ret pad(cv, str::from_char(c), pad_nozero);
    }
    fn conv_str(cv: &conv, s: str) -> str {
        // For strings, precision is the maximum characters
        // displayed

        // FIXME: substr works on bytes, not chars!
        let unpadded =
            alt cv.precision {
              count_implied. { s }
              count_is(max) {
                if max as uint < str::char_len(s) {
                    str::substr(s, 0u, max as uint)
                } else { s }
              }
            };
        ret pad(cv, unpadded, pad_nozero);
    }

    // Convert an int to string with minimum number of digits. If precision is
    // 0 and num is 0 then the result is the empty string.
    fn int_to_str_prec(num: int, radix: uint, prec: uint) -> str {
        ret if num < 0 {
                "-" + uint_to_str_prec(-num as uint, radix, prec)
            } else { uint_to_str_prec(num as uint, radix, prec) };
    }

    // Convert a uint to string with a minimum number of digits.  If precision
    // is 0 and num is 0 then the result is the empty string. Could move this
    // to uint: but it doesn't seem all that useful.
    fn uint_to_str_prec(num: uint, radix: uint, prec: uint) -> str {
        ret if prec == 0u && num == 0u {
                ""
            } else {
                let s = uint::to_str(num, radix);
                let len = str::char_len(s);
                if len < prec {
                    let diff = prec - len;
                    let pad = str_init_elt('0', diff);
                    pad + s
                } else { s }
            };
    }
    fn get_int_precision(cv: &conv) -> uint {
        ret alt cv.precision {
              count_is(c) { c as uint }
              count_implied. { 1u }
            };
    }

    // FIXME: This might be useful in str: but needs to be utf8 safe first
    fn str_init_elt(c: char, n_elts: uint) -> str {
        let svec = vec::init_elt::<u8>(c as u8, n_elts);

        ret str::unsafe_from_bytes(svec);
    }
    tag pad_mode { pad_signed; pad_unsigned; pad_nozero; }
    fn pad(cv: &conv, s: str, mode: pad_mode) -> str {
        let uwidth;
        alt cv.width {
          count_implied. { ret s; }
          count_is(width) {
            // FIXME: Maybe width should be uint

            uwidth = width as uint;
          }
        }
        let strlen = str::char_len(s);
        if uwidth <= strlen { ret s; }
        let padchar = ' ';
        let diff = uwidth - strlen;
        if have_flag(cv.flags, flag_left_justify) {
            let padstr = str_init_elt(padchar, diff);
            ret s + padstr;
        }
        let might_zero_pad = false;
        let signed = false;
        alt mode {
          pad_nozero. {
            // fallthrough

          }
          pad_signed. { might_zero_pad = true; signed = true; }
          pad_unsigned. { might_zero_pad = true; }
        }
        fn have_precision(cv: &conv) -> bool {
            ret alt cv.precision { count_implied. { false } _ { true } };
        }
        let zero_padding = false;
        if might_zero_pad && have_flag(cv.flags, flag_left_zero_pad) &&
               !have_precision(cv) {
            padchar = '0';
            zero_padding = true;
        }
        let padstr = str_init_elt(padchar, diff);
        // This is completely heinous. If we have a signed value then
        // potentially rip apart the intermediate result and insert some
        // zeros. It may make sense to convert zero padding to a precision
        // instead.

        if signed && zero_padding && str::byte_len(s) > 0u {
            let head = s.(0);
            if head == '+' as u8 || head == '-' as u8 || head == ' ' as u8 {
                let headstr = str::unsafe_from_bytes(~[head]);
                let bytelen = str::byte_len(s);
                let numpart = str::substr(s, 1u, bytelen - 1u);
                ret headstr + padstr + numpart;
            }
        }
        ret padstr + s;
    }
    fn have_flag(flags: &[flag], f: flag) -> bool {
        for candidate: flag in flags { if candidate == f { ret true; } }
        ret false;
    }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
