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

use cmp::Eq;
use option::{Some, None};


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
    #[legacy_exports];
    enum Signedness { Signed, Unsigned, }
    enum Caseness { CaseUpper, CaseLower, }
    enum Ty {
        TyBool,
        TyStr,
        TyChar,
        TyInt(Signedness),
        TyBits,
        TyHex(Caseness),
        TyOctal,
        TyFloat,
        TyPoly,
    }
    enum Flag {
        FlagLeftJustify,
        FlagLeftZeroPad,
        FlagSpaceForSign,
        FlagSignAlways,
        FlagAlternate,
    }
    enum Count {
        CountIs(int),
        CountIsParam(int),
        CountIsNextParam,
        CountImplied,
    }

    // A formatted conversion from an expression to a string
    type Conv =
        {param: Option<int>,
         flags: ~[Flag],
         width: Count,
         precision: Count,
         ty: Ty};


    // A fragment of the output sequence
    enum Piece { PieceString(~str), PieceConv(Conv), }
    type ErrorFn = fn@(&str) -> ! ;

    fn parse_fmt_string(s: &str, error: ErrorFn) -> ~[Piece] {
        let mut pieces: ~[Piece] = ~[];
        let lim = str::len(s);
        let mut buf = ~"";
        fn flush_buf(buf: ~str, pieces: &mut ~[Piece]) -> ~str {
            if buf.len() > 0 {
                let piece = PieceString(move buf);
                pieces.push(move piece);
            }
            return ~"";
        }
        let mut i = 0;
        while i < lim {
            let size = str::utf8_char_width(s[i]);
            let curr = str::slice(s, i, i+size);
            if curr == ~"%" {
                i += 1;
                if i >= lim {
                    error(~"unterminated conversion at end of string");
                }
                let curr2 = str::slice(s, i, i+1);
                if curr2 == ~"%" {
                    buf += curr2;
                    i += 1;
                } else {
                    buf = flush_buf(move buf, &mut pieces);
                    let rs = parse_conversion(s, i, lim, error);
                    pieces.push(copy rs.piece);
                    i = rs.next;
                }
            } else { buf += curr; i += size; }
        }
        flush_buf(move buf, &mut pieces);
        move pieces
    }
    fn peek_num(s: &str, i: uint, lim: uint) ->
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
    fn parse_conversion(s: &str, i: uint, lim: uint, error: ErrorFn) ->
       {piece: Piece, next: uint} {
        let parm = parse_parameter(s, i, lim);
        let flags = parse_flags(s, parm.next, lim);
        let width = parse_count(s, flags.next, lim);
        let prec = parse_precision(s, width.next, lim);
        let ty = parse_type(s, prec.next, lim, error);
        return {piece:
                 PieceConv({param: parm.param,
                             flags: copy flags.flags,
                             width: width.count,
                             precision: prec.count,
                             ty: ty.ty}),
             next: ty.next};
    }
    fn parse_parameter(s: &str, i: uint, lim: uint) ->
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
    fn parse_flags(s: &str, i: uint, lim: uint) ->
       {flags: ~[Flag], next: uint} {
        let noflags: ~[Flag] = ~[];
        if i >= lim { return {flags: move noflags, next: i}; }

        fn more(f: Flag, s: &str, i: uint, lim: uint) ->
           {flags: ~[Flag], next: uint} {
            let next = parse_flags(s, i + 1u, lim);
            let rest = copy next.flags;
            let j = next.next;
            let curr: ~[Flag] = ~[f];
            return {flags: vec::append(move curr, rest), next: j};
        }
        // Unfortunate, but because s is borrowed, can't use a closure
     //   fn more(f: Flag, s: &str) { more_(f, s, i, lim); }
        let f = s[i];
        return if f == '-' as u8 {
                more(FlagLeftJustify, s, i, lim)
            } else if f == '0' as u8 {
                more(FlagLeftZeroPad, s, i, lim)
            } else if f == ' ' as u8 {
                more(FlagSpaceForSign, s, i, lim)
            } else if f == '+' as u8 {
                more(FlagSignAlways, s, i, lim)
            } else if f == '#' as u8 {
                more(FlagAlternate, s, i, lim)
            } else { {flags: move noflags, next: i} };
    }
    fn parse_count(s: &str, i: uint, lim: uint)
        -> {count: Count, next: uint} {
        return if i >= lim {
                {count: CountImplied, next: i}
            } else if s[i] == '*' as u8 {
                let param = parse_parameter(s, i + 1u, lim);
                let j = param.next;
                match param.param {
                  None => {count: CountIsNextParam, next: j},
                  Some(n) => {count: CountIsParam(n), next: j}
                }
            } else {
                let num = peek_num(s, i, lim);
                match num {
                  None => {count: CountImplied, next: i},
                  Some(num) => {
                    count: CountIs(num.num as int),
                    next: num.next
                  }
                }
            };
    }
    fn parse_precision(s: &str, i: uint, lim: uint) ->
       {count: Count, next: uint} {
        return if i >= lim {
                {count: CountImplied, next: i}
            } else if s[i] == '.' as u8 {
                let count = parse_count(s, i + 1u, lim);


                // If there were no digits specified, i.e. the precision
                // was ".", then the precision is 0
                match count.count {
                  CountImplied => {count: CountIs(0), next: count.next},
                  _ => count
                }
            } else { {count: CountImplied, next: i} };
    }
    fn parse_type(s: &str, i: uint, lim: uint, error: ErrorFn) ->
       {ty: Ty, next: uint} {
        if i >= lim { error(~"missing type in conversion"); }
        let tstr = str::slice(s, i, i+1u);
        // FIXME (#2249): Do we really want two signed types here?
        // How important is it to be printf compatible?
        let t =
            if tstr == ~"b" {
                TyBool
            } else if tstr == ~"s" {
                TyStr
            } else if tstr == ~"c" {
                TyChar
            } else if tstr == ~"d" || tstr == ~"i" {
                TyInt(Signed)
            } else if tstr == ~"u" {
                TyInt(Unsigned)
            } else if tstr == ~"x" {
                TyHex(CaseLower)
            } else if tstr == ~"X" {
                TyHex(CaseUpper)
            } else if tstr == ~"t" {
                TyBits
            } else if tstr == ~"o" {
                TyOctal
            } else if tstr == ~"f" {
                TyFloat
            } else if tstr == ~"?" {
                TyPoly
            } else { error(~"unknown type in conversion: " + tstr) };
        return {ty: t, next: i + 1u};
    }
}

// Functions used by the fmt extension at runtime. For now there are a lot of
// decisions made a runtime. If it proves worthwhile then some of these
// conditions can be evaluated at compile-time. For now though it's cleaner to
// implement it 0this way, I think.
mod rt {
    #[legacy_exports];
    const flag_none : u32 = 0u32;
    const flag_left_justify   : u32 = 0b00000000000000000000000000000001u32;
    const flag_left_zero_pad  : u32 = 0b00000000000000000000000000000010u32;
    const flag_space_for_sign : u32 = 0b00000000000000000000000000000100u32;
    const flag_sign_always    : u32 = 0b00000000000000000000000000001000u32;
    const flag_alternate      : u32 = 0b00000000000000000000000000010000u32;

    enum Count { CountIs(int), CountImplied, }
    enum Ty { TyDefault, TyBits, TyHexUpper, TyHexLower, TyOctal, }

    type Conv = {flags: u32, width: Count, precision: Count, ty: Ty};

    pure fn conv_int(cv: Conv, i: int) -> ~str {
        let radix = 10;
        let prec = get_int_precision(cv);
        let mut s : ~str = int_to_str_prec(i, radix, prec);
        if 0 <= i {
            if have_flag(cv.flags, flag_sign_always) {
                unsafe { str::unshift_char(&mut s, '+') };
            } else if have_flag(cv.flags, flag_space_for_sign) {
                unsafe { str::unshift_char(&mut s, ' ') };
            }
        }
        return unsafe { pad(cv, s, PadSigned) };
    }
    pure fn conv_uint(cv: Conv, u: uint) -> ~str {
        let prec = get_int_precision(cv);
        let mut rs =
            match cv.ty {
              TyDefault => uint_to_str_prec(u, 10u, prec),
              TyHexLower => uint_to_str_prec(u, 16u, prec),
              TyHexUpper => str::to_upper(uint_to_str_prec(u, 16u, prec)),
              TyBits => uint_to_str_prec(u, 2u, prec),
              TyOctal => uint_to_str_prec(u, 8u, prec)
            };
        return unsafe { pad(cv, rs, PadUnsigned) };
    }
    pure fn conv_bool(cv: Conv, b: bool) -> ~str {
        let s = if b { ~"true" } else { ~"false" };
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        return conv_str(cv, s);
    }
    pure fn conv_char(cv: Conv, c: char) -> ~str {
        let mut s = str::from_char(c);
        return unsafe { pad(cv, s, PadNozero) };
    }
    pure fn conv_str(cv: Conv, s: &str) -> ~str {
        // For strings, precision is the maximum characters
        // displayed
        let mut unpadded = match cv.precision {
          CountImplied => s.to_unique(),
          CountIs(max) => if max as uint < str::char_len(s) {
            str::substr(s, 0u, max as uint)
          } else {
            s.to_unique()
          }
        };
        return unsafe { pad(cv, unpadded, PadNozero) };
    }
    pure fn conv_float(cv: Conv, f: float) -> ~str {
        let (to_str, digits) = match cv.precision {
              CountIs(c) => (float::to_str_exact, c as uint),
              CountImplied => (float::to_str, 6u)
        };
        let mut s = unsafe { to_str(f, digits) };
        if 0.0 <= f {
            if have_flag(cv.flags, flag_sign_always) {
                s = ~"+" + s;
            } else if have_flag(cv.flags, flag_space_for_sign) {
                s = ~" " + s;
            }
        }
        return unsafe { pad(cv, s, PadFloat) };
    }
    pure fn conv_poly<T>(cv: Conv, v: &T) -> ~str {
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
                } else { move s }
            };
    }
    pure fn get_int_precision(cv: Conv) -> uint {
        return match cv.precision {
              CountIs(c) => c as uint,
              CountImplied => 1u
            };
    }

    enum PadMode { PadSigned, PadUnsigned, PadNozero, PadFloat }

    impl PadMode : Eq {
        pure fn eq(other: &PadMode) -> bool {
            match (self, (*other)) {
                (PadSigned, PadSigned) => true,
                (PadUnsigned, PadUnsigned) => true,
                (PadNozero, PadNozero) => true,
                (PadFloat, PadFloat) => true,
                (PadSigned, _) => false,
                (PadUnsigned, _) => false,
                (PadNozero, _) => false,
                (PadFloat, _) => false
            }
        }
        pure fn ne(other: &PadMode) -> bool { !self.eq(other) }
    }

    fn pad(cv: Conv, s: ~str, mode: PadMode) -> ~str {
        let mut s = move s; // sadtimes
        let uwidth : uint = match cv.width {
          CountImplied => return s,
          CountIs(width) => {
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
          PadNozero => {might_zero_pad:false, signed:false},
          PadSigned => {might_zero_pad:true,  signed:true },
          PadFloat => {might_zero_pad:true,  signed:true},
          PadUnsigned => {might_zero_pad:true,  signed:false}
        };
        pure fn have_precision(cv: Conv) -> bool {
            return match cv.precision { CountImplied => false, _ => true };
        }
        let zero_padding = {
            if might_zero_pad && have_flag(cv.flags, flag_left_zero_pad) &&
                (!have_precision(cv) || mode == PadFloat) {
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

        if signed && zero_padding && s.len() > 0 {
            let head = str::shift_char(&mut s);
            if head == '+' || head == '-' || head == ' ' {
                let headstr = str::from_chars(vec::from_elem(1u, head));
                return headstr + padstr + s;
            }
            else {
                str::unshift_char(&mut s, head);
            }
        }
        return padstr + s;
    }
    pure fn have_flag(flags: u32, f: u32) -> bool {
        flags & f != 0
    }
}

// Remove after snapshot

// Functions used by the fmt extension at runtime. For now there are a lot of
// decisions made a runtime. If it proves worthwhile then some of these
// conditions can be evaluated at compile-time. For now though it's cleaner to
// implement it 0this way, I think.
mod rt2 {
    #[legacy_exports];
    const flag_none : u32 = 0u32;
    const flag_left_justify   : u32 = 0b00000000000000000000000000000001u32;
    const flag_left_zero_pad  : u32 = 0b00000000000000000000000000000010u32;
    const flag_space_for_sign : u32 = 0b00000000000000000000000000000100u32;
    const flag_sign_always    : u32 = 0b00000000000000000000000000001000u32;
    const flag_alternate      : u32 = 0b00000000000000000000000000010000u32;

    enum Count { CountIs(int), CountImplied, }
    enum Ty { TyDefault, TyBits, TyHexUpper, TyHexLower, TyOctal, }

    type Conv = {flags: u32, width: Count, precision: Count, ty: Ty};

    pure fn conv_int(cv: Conv, i: int) -> ~str {
        let radix = 10;
        let prec = get_int_precision(cv);
        let mut s : ~str = int_to_str_prec(i, radix, prec);
        if 0 <= i {
            if have_flag(cv.flags, flag_sign_always) {
                unsafe { str::unshift_char(&mut s, '+') };
            } else if have_flag(cv.flags, flag_space_for_sign) {
                unsafe { str::unshift_char(&mut s, ' ') };
            }
        }
        return unsafe { pad(cv, s, PadSigned) };
    }
    pure fn conv_uint(cv: Conv, u: uint) -> ~str {
        let prec = get_int_precision(cv);
        let mut rs =
            match cv.ty {
              TyDefault => uint_to_str_prec(u, 10u, prec),
              TyHexLower => uint_to_str_prec(u, 16u, prec),
              TyHexUpper => str::to_upper(uint_to_str_prec(u, 16u, prec)),
              TyBits => uint_to_str_prec(u, 2u, prec),
              TyOctal => uint_to_str_prec(u, 8u, prec)
            };
        return unsafe { pad(cv, rs, PadUnsigned) };
    }
    pure fn conv_bool(cv: Conv, b: bool) -> ~str {
        let s = if b { ~"true" } else { ~"false" };
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        return conv_str(cv, s);
    }
    pure fn conv_char(cv: Conv, c: char) -> ~str {
        let mut s = str::from_char(c);
        return unsafe { pad(cv, s, PadNozero) };
    }
    pure fn conv_str(cv: Conv, s: &str) -> ~str {
        // For strings, precision is the maximum characters
        // displayed
        let mut unpadded = match cv.precision {
          CountImplied => s.to_unique(),
          CountIs(max) => if max as uint < str::char_len(s) {
            str::substr(s, 0u, max as uint)
          } else {
            s.to_unique()
          }
        };
        return unsafe { pad(cv, unpadded, PadNozero) };
    }
    pure fn conv_float(cv: Conv, f: float) -> ~str {
        let (to_str, digits) = match cv.precision {
              CountIs(c) => (float::to_str_exact, c as uint),
              CountImplied => (float::to_str, 6u)
        };
        let mut s = unsafe { to_str(f, digits) };
        if 0.0 <= f {
            if have_flag(cv.flags, flag_sign_always) {
                s = ~"+" + s;
            } else if have_flag(cv.flags, flag_space_for_sign) {
                s = ~" " + s;
            }
        }
        return unsafe { pad(cv, s, PadFloat) };
    }
    pure fn conv_poly<T>(cv: Conv, v: &T) -> ~str {
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
                } else { move s }
            };
    }
    pure fn get_int_precision(cv: Conv) -> uint {
        return match cv.precision {
              CountIs(c) => c as uint,
              CountImplied => 1u
            };
    }

    enum PadMode { PadSigned, PadUnsigned, PadNozero, PadFloat }

    impl PadMode : Eq {
        pure fn eq(other: &PadMode) -> bool {
            match (self, (*other)) {
                (PadSigned, PadSigned) => true,
                (PadUnsigned, PadUnsigned) => true,
                (PadNozero, PadNozero) => true,
                (PadFloat, PadFloat) => true,
                (PadSigned, _) => false,
                (PadUnsigned, _) => false,
                (PadNozero, _) => false,
                (PadFloat, _) => false
            }
        }
        pure fn ne(other: &PadMode) -> bool { !self.eq(other) }
    }

    fn pad(cv: Conv, s: ~str, mode: PadMode) -> ~str {
        let mut s = move s; // sadtimes
        let uwidth : uint = match cv.width {
          CountImplied => return s,
          CountIs(width) => {
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
          PadNozero => {might_zero_pad:false, signed:false},
          PadSigned => {might_zero_pad:true,  signed:true },
          PadFloat => {might_zero_pad:true,  signed:true},
          PadUnsigned => {might_zero_pad:true,  signed:false}
        };
        pure fn have_precision(cv: Conv) -> bool {
            return match cv.precision { CountImplied => false, _ => true };
        }
        let zero_padding = {
            if might_zero_pad && have_flag(cv.flags, flag_left_zero_pad) &&
                (!have_precision(cv) || mode == PadFloat) {
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

        if signed && zero_padding && s.len() > 0 {
            let head = str::shift_char(&mut s);
            if head == '+' || head == '-' || head == ' ' {
                let headstr = str::from_chars(vec::from_elem(1u, head));
                return headstr + padstr + s;
            }
            else {
                str::unshift_char(&mut s, head);
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
    #[legacy_exports];
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
