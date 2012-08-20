#[doc(hidden)];

/*
Syntax Extension: fmt

Format a string

The 'fmt' extension is modeled on the posix printf system.

A posix conversion ostensibly looks like this

> %~[parameter]~[flags]~[width]~[.precision]~[length]type

Given the different numeric type bestiary we have, we omit the 'length'
parameter and support slightly different conversions for 'type'

> %~[parameter]~[flags]~[width]~[.precision]type

we also only support translating-to-rust a tiny subset of the possible
combinations at the moment.

Example:

debug!("hello, %s!", "world");

*/

import option::{Some, None};


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
    enum signedness { signed, unsigned, }
    enum caseness { case_upper, case_lower, }
    enum ty {
        ty_bool,
        ty_str,
        ty_char,
        ty_int(signedness),
        ty_bits,
        ty_hex(caseness),
        ty_octal,
        ty_float,
        ty_poly,
    }
    enum flag {
        flag_left_justify,
        flag_left_zero_pad,
        flag_space_for_sign,
        flag_sign_always,
        flag_alternate,
    }
    enum count {
        count_is(int),
        count_is_param(int),
        count_is_next_param,
        count_implied,
    }

    // A formatted conversion from an expression to a string
    type conv =
        {param: Option<int>,
         flags: ~[flag],
         width: count,
         precision: count,
         ty: ty};


    // A fragment of the output sequence
    enum piece { piece_string(~str), piece_conv(conv), }
    type error_fn = fn@(~str) -> ! ;

    fn parse_fmt_string(s: ~str, error: error_fn) -> ~[piece] {
        let mut pieces: ~[piece] = ~[];
        let lim = str::len(s);
        let mut buf = ~"";
        fn flush_buf(buf: ~str, &pieces: ~[piece]) -> ~str {
            if str::len(buf) > 0u {
                let piece = piece_string(buf);
                vec::push(pieces, piece);
            }
            return ~"";
        }
        let mut i = 0u;
        while i < lim {
            let size = str::utf8_char_width(s[i]);
            let curr = str::slice(s, i, i+size);
            if curr == ~"%" {
                i += 1u;
                if i >= lim {
                    error(~"unterminated conversion at end of string");
                }
                let curr2 = str::slice(s, i, i+1u);
                if curr2 == ~"%" {
                    buf += curr2;
                    i += 1u;
                } else {
                    buf = flush_buf(buf, pieces);
                    let rs = parse_conversion(s, i, lim, error);
                    vec::push(pieces, rs.piece);
                    i = rs.next;
                }
            } else { buf += curr; i += size; }
        }
        flush_buf(buf, pieces);
        return pieces;
    }
    fn peek_num(s: ~str, i: uint, lim: uint) ->
       Option<{num: uint, next: uint}> {
        let mut j = i;
        let mut accum = 0u;
        let mut found = false;
        while j < lim {
            match char::to_digit(s[j] as char, 10) {
                Some(x) => {
                    found = true;
                    accum *= 10;
                    accum += x;
                    j += 1;
                },
                None => break
            }
        }
        if found {
            Some({num: accum, next: j})
        } else {
            None
        }
    }
    fn parse_conversion(s: ~str, i: uint, lim: uint, error: error_fn) ->
       {piece: piece, next: uint} {
        let parm = parse_parameter(s, i, lim);
        let flags = parse_flags(s, parm.next, lim);
        let width = parse_count(s, flags.next, lim);
        let prec = parse_precision(s, width.next, lim);
        let ty = parse_type(s, prec.next, lim, error);
        return {piece:
                 piece_conv({param: parm.param,
                             flags: flags.flags,
                             width: width.count,
                             precision: prec.count,
                             ty: ty.ty}),
             next: ty.next};
    }
    fn parse_parameter(s: ~str, i: uint, lim: uint) ->
       {param: Option<int>, next: uint} {
        if i >= lim { return {param: None, next: i}; }
        let num = peek_num(s, i, lim);
        return match num {
              None => {param: None, next: i},
              Some(t) => {
                let n = t.num;
                let j = t.next;
                if j < lim && s[j] == '$' as u8 {
                    {param: Some(n as int), next: j + 1u}
                } else { {param: None, next: i} }
              }
            };
    }
    fn parse_flags(s: ~str, i: uint, lim: uint) ->
       {flags: ~[flag], next: uint} {
        let noflags: ~[flag] = ~[];
        if i >= lim { return {flags: noflags, next: i}; }

        fn more_(f: flag, s: ~str, i: uint, lim: uint) ->
           {flags: ~[flag], next: uint} {
            let next = parse_flags(s, i + 1u, lim);
            let rest = next.flags;
            let j = next.next;
            let curr: ~[flag] = ~[f];
            return {flags: vec::append(curr, rest), next: j};
        }
        let more = |x| more_(x, s, i, lim);
        let f = s[i];
        return if f == '-' as u8 {
                more(flag_left_justify)
            } else if f == '0' as u8 {
                more(flag_left_zero_pad)
            } else if f == ' ' as u8 {
                more(flag_space_for_sign)
            } else if f == '+' as u8 {
                more(flag_sign_always)
            } else if f == '#' as u8 {
                more(flag_alternate)
            } else { {flags: noflags, next: i} };
    }
    fn parse_count(s: ~str, i: uint, lim: uint)
        -> {count: count, next: uint} {
        return if i >= lim {
                {count: count_implied, next: i}
            } else if s[i] == '*' as u8 {
                let param = parse_parameter(s, i + 1u, lim);
                let j = param.next;
                match param.param {
                  None => {count: count_is_next_param, next: j},
                  Some(n) => {count: count_is_param(n), next: j}
                }
            } else {
                let num = peek_num(s, i, lim);
                match num {
                  None => {count: count_implied, next: i},
                  Some(num) => {
                    count: count_is(num.num as int),
                    next: num.next
                  }
                }
            };
    }
    fn parse_precision(s: ~str, i: uint, lim: uint) ->
       {count: count, next: uint} {
        return if i >= lim {
                {count: count_implied, next: i}
            } else if s[i] == '.' as u8 {
                let count = parse_count(s, i + 1u, lim);


                // If there were no digits specified, i.e. the precision
                // was ".", then the precision is 0
                match count.count {
                  count_implied => {count: count_is(0), next: count.next},
                  _ => count
                }
            } else { {count: count_implied, next: i} };
    }
    fn parse_type(s: ~str, i: uint, lim: uint, error: error_fn) ->
       {ty: ty, next: uint} {
        if i >= lim { error(~"missing type in conversion"); }
        let tstr = str::slice(s, i, i+1u);
        // FIXME (#2249): Do we really want two signed types here?
        // How important is it to be printf compatible?
        let t =
            if tstr == ~"b" {
                ty_bool
            } else if tstr == ~"s" {
                ty_str
            } else if tstr == ~"c" {
                ty_char
            } else if tstr == ~"d" || tstr == ~"i" {
                ty_int(signed)
            } else if tstr == ~"u" {
                ty_int(unsigned)
            } else if tstr == ~"x" {
                ty_hex(case_lower)
            } else if tstr == ~"X" {
                ty_hex(case_upper)
            } else if tstr == ~"t" {
                ty_bits
            } else if tstr == ~"o" {
                ty_octal
            } else if tstr == ~"f" {
                ty_float
            } else if tstr == ~"?" {
                ty_poly
            } else { error(~"unknown type in conversion: " + tstr) };
        return {ty: t, next: i + 1u};
    }
}


// Functions used by the fmt extension at runtime. For now there are a lot of
// decisions made a runtime. If it proves worthwhile then some of these
// conditions can be evaluated at compile-time. For now though it's cleaner to
// implement it 0this way, I think.
mod rt {
    const flag_none : u32 = 0u32;
    const flag_left_justify   : u32 = 0b00000000000000000000000000000001u32;
    const flag_left_zero_pad  : u32 = 0b00000000000000000000000000000010u32;
    const flag_space_for_sign : u32 = 0b00000000000000000000000000000100u32;
    const flag_sign_always    : u32 = 0b00000000000000000000000000001000u32;
    const flag_alternate      : u32 = 0b00000000000000000000000000010000u32;

    enum count { count_is(int), count_implied, }
    enum ty { ty_default, ty_bits, ty_hex_upper, ty_hex_lower, ty_octal, }

    type conv = {flags: u32, width: count, precision: count, ty: ty};

    pure fn conv_int(cv: conv, i: int) -> ~str {
        let radix = 10u;
        let prec = get_int_precision(cv);
        let mut s : ~str = int_to_str_prec(i, radix, prec);
        if 0 <= i {
            if have_flag(cv.flags, flag_sign_always) {
                unchecked { str::unshift_char(s, '+') };
            } else if have_flag(cv.flags, flag_space_for_sign) {
                unchecked { str::unshift_char(s, ' ') };
            }
        }
        return unchecked { pad(cv, s, pad_signed) };
    }
    pure fn conv_uint(cv: conv, u: uint) -> ~str {
        let prec = get_int_precision(cv);
        let mut rs =
            match cv.ty {
              ty_default => uint_to_str_prec(u, 10u, prec),
              ty_hex_lower => uint_to_str_prec(u, 16u, prec),
              ty_hex_upper => str::to_upper(uint_to_str_prec(u, 16u, prec)),
              ty_bits => uint_to_str_prec(u, 2u, prec),
              ty_octal => uint_to_str_prec(u, 8u, prec)
            };
        return unchecked { pad(cv, rs, pad_unsigned) };
    }
    pure fn conv_bool(cv: conv, b: bool) -> ~str {
        let s = if b { ~"true" } else { ~"false" };
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        return conv_str(cv, s);
    }
    pure fn conv_char(cv: conv, c: char) -> ~str {
        let mut s = str::from_char(c);
        return unchecked { pad(cv, s, pad_nozero) };
    }
    pure fn conv_str(cv: conv, s: &str) -> ~str {
        // For strings, precision is the maximum characters
        // displayed
        let mut unpadded = match cv.precision {
          count_implied => s.to_unique(),
          count_is(max) => if max as uint < str::char_len(s) {
            str::substr(s, 0u, max as uint)
          } else {
            s.to_unique()
          }
        };
        return unchecked { pad(cv, unpadded, pad_nozero) };
    }
    pure fn conv_float(cv: conv, f: float) -> ~str {
        let (to_str, digits) = match cv.precision {
              count_is(c) => (float::to_str_exact, c as uint),
              count_implied => (float::to_str, 6u)
        };
        let mut s = unchecked { to_str(f, digits) };
        if 0.0 <= f {
            if have_flag(cv.flags, flag_sign_always) {
                s = ~"+" + s;
            } else if have_flag(cv.flags, flag_space_for_sign) {
                s = ~" " + s;
            }
        }
        return unchecked { pad(cv, s, pad_float) };
    }
    pure fn conv_poly<T>(cv: conv, v: T) -> ~str {
        let s = sys::log_str(v);
        return conv_str(cv, s);
    }

    // Convert an int to string with minimum number of digits. If precision is
    // 0 and num is 0 then the result is the empty string.
    pure fn int_to_str_prec(num: int, radix: uint, prec: uint) -> ~str {
        return if num < 0 {
                ~"-" + uint_to_str_prec(-num as uint, radix, prec)
            } else { uint_to_str_prec(num as uint, radix, prec) };
    }

    // Convert a uint to string with a minimum number of digits.  If precision
    // is 0 and num is 0 then the result is the empty string. Could move this
    // to uint: but it doesn't seem all that useful.
    pure fn uint_to_str_prec(num: uint, radix: uint, prec: uint) -> ~str {
        return if prec == 0u && num == 0u {
                ~""
            } else {
                let s = uint::to_str(num, radix);
                let len = str::char_len(s);
                if len < prec {
                    let diff = prec - len;
                    let pad = str::from_chars(vec::from_elem(diff, '0'));
                    pad + s
                } else { s }
            };
    }
    pure fn get_int_precision(cv: conv) -> uint {
        return match cv.precision {
              count_is(c) => c as uint,
              count_implied => 1u
            };
    }
    enum pad_mode { pad_signed, pad_unsigned, pad_nozero, pad_float }
    fn pad(cv: conv, &s: ~str, mode: pad_mode) -> ~str {
        let uwidth : uint = match cv.width {
          count_implied => return s,
          count_is(width) => {
              // FIXME: width should probably be uint (see Issue #1996)
              width as uint
          }
        };
        let strlen = str::char_len(s);
        if uwidth <= strlen { return s; }
        let mut padchar = ' ';
        let diff = uwidth - strlen;
        if have_flag(cv.flags, flag_left_justify) {
            let padstr = str::from_chars(vec::from_elem(diff, padchar));
            return s + padstr;
        }
        let {might_zero_pad, signed} = match mode {
          pad_nozero => {might_zero_pad:false, signed:false},
          pad_signed => {might_zero_pad:true,  signed:true },
          pad_float => {might_zero_pad:true,  signed:true},
          pad_unsigned => {might_zero_pad:true,  signed:false}
        };
        pure fn have_precision(cv: conv) -> bool {
            return match cv.precision { count_implied => false, _ => true };
        }
        let zero_padding = {
            if might_zero_pad && have_flag(cv.flags, flag_left_zero_pad) &&
                (!have_precision(cv) || mode == pad_float) {
                padchar = '0';
                true
            } else {
                false
            }
        };
        let padstr = str::from_chars(vec::from_elem(diff, padchar));
        // This is completely heinous. If we have a signed value then
        // potentially rip apart the intermediate result and insert some
        // zeros. It may make sense to convert zero padding to a precision
        // instead.

        if signed && zero_padding && str::len(s) > 0u {
            let head = str::shift_char(s);
            if head == '+' || head == '-' || head == ' ' {
                let headstr = str::from_chars(vec::from_elem(1u, head));
                return headstr + padstr + s;
            }
            else {
                str::unshift_char(s, head);
            }
        }
        return padstr + s;
    }
    pure fn have_flag(flags: u32, f: u32) -> bool {
        flags & f != 0
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn fmt_slice() {
        let s = "abc";
        let _s = fmt!("%s", s);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
