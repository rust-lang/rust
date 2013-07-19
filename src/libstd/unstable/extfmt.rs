// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support for fmt! expressions.
//!
//! The syntax is close to that of Posix format strings:
//!
//! ~~~~~~
//! Format := '%' Parameter? Flag* Width? Precision? Type
//! Parameter := [0-9]+ '$'
//! Flag := [ 0#+-]
//! Width := Parameter | [0-9]+
//! Precision := '.' [0-9]+
//! Type := [bcdfiostuxX?]
//! ~~~~~~
//!
//! * Parameter is the 1-based argument to apply the format to. Currently not
//! implemented.
//! * Flag 0 causes leading zeros to be used for padding when converting
//! numbers.
//! * Flag # causes the conversion to be done in an *alternative* manner.
//! Currently not implemented.
//! * Flag + causes signed numbers to always be prepended with a sign
//! character.
//! * Flag - left justifies the result
//! * Width specifies the minimum field width of the result. By default
//! leading spaces are added.
//! * Precision specifies the minimum number of digits for integral types
//! and the minimum number
//! of decimal places for float.
//!
//! The types currently supported are:
//!
//! * b - bool
//! * c - char
//! * d - int
//! * f - float
//! * i - int (same as d)
//! * o - uint as octal
//! * t - uint as binary
//! * u - uint
//! * x - uint as lower-case hexadecimal
//! * X - uint as upper-case hexadecimal
//! * s - str (any flavor)
//! * ? - arbitrary type (does not use the to_str trait)

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

use prelude::*;
use iterator::IteratorUtil;

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
#[doc(hidden)]
pub mod ct {
    use char;
    use container::Container;
    use prelude::*;
    use str;

    #[deriving(Eq)]
    pub enum Signedness { Signed, Unsigned, }

    #[deriving(Eq)]
    pub enum Caseness { CaseUpper, CaseLower, }

    #[deriving(Eq)]
    pub enum Ty {
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

    #[deriving(Eq)]
    pub enum Flag {
        FlagLeftJustify,
        FlagLeftZeroPad,
        FlagSpaceForSign,
        FlagSignAlways,
        FlagAlternate,
    }

    #[deriving(Eq)]
    pub enum Count {
        CountIs(uint),
        CountIsParam(uint),
        CountIsNextParam,
        CountImplied,
    }

    #[deriving(Eq)]
    struct Parsed<T> {
        val: T,
        next: uint
    }

    impl<T> Parsed<T> {
        pub fn new(val: T, next: uint) -> Parsed<T> {
            Parsed {val: val, next: next}
        }
    }

    // A formatted conversion from an expression to a string
    #[deriving(Eq)]
    pub struct Conv {
        param: Option<uint>,
        flags: ~[Flag],
        width: Count,
        precision: Count,
        ty: Ty
    }

    // A fragment of the output sequence
    #[deriving(Eq)]
    pub enum Piece { PieceString(~str), PieceConv(Conv), }

    pub type ErrorFn = @fn(&str) -> !;

    pub fn parse_fmt_string(s: &str, err: ErrorFn) -> ~[Piece] {
        fn push_slice(ps: &mut ~[Piece], s: &str, from: uint, to: uint) {
            if to > from {
                ps.push(PieceString(s.slice(from, to).to_owned()));
            }
        }

        let lim = s.len();
        let mut h = 0;
        let mut i = 0;
        let mut pieces = ~[];

        while i < lim {
            if s[i] == '%' as u8 {
                i += 1;

                if i >= lim {
                    err("unterminated conversion at end of string");
                } else if s[i] == '%' as u8 {
                    push_slice(&mut pieces, s, h, i);
                    i += 1;
                } else {
                    push_slice(&mut pieces, s, h, i - 1);
                    let Parsed {val, next} = parse_conversion(s, i, lim, err);
                    pieces.push(val);
                    i = next;
                }

                h = i;
            } else {
                i += str::utf8_char_width(s[i]);
            }
        }

        push_slice(&mut pieces, s, h, i);
        pieces
    }

    pub fn peek_num(s: &str, i: uint, lim: uint) -> Option<Parsed<uint>> {
        let mut i = i;
        let mut accum = 0;
        let mut found = false;

        while i < lim {
            match char::to_digit(s[i] as char, 10) {
                Some(x) => {
                    found = true;
                    accum *= 10;
                    accum += x;
                    i += 1;
                }
                None => break
            }
        }

        if found {
            Some(Parsed::new(accum, i))
        } else {
            None
        }
    }

    pub fn parse_conversion(s: &str, i: uint, lim: uint, err: ErrorFn) ->
        Parsed<Piece> {
        let param = parse_parameter(s, i, lim);
        // avoid copying ~[Flag] by destructuring
        let Parsed {val: flags_val, next: flags_next} = parse_flags(s,
            param.next, lim);
        let width = parse_count(s, flags_next, lim);
        let prec = parse_precision(s, width.next, lim);
        let ty = parse_type(s, prec.next, lim, err);

        Parsed::new(PieceConv(Conv {
            param: param.val,
            flags: flags_val,
            width: width.val,
            precision: prec.val,
            ty: ty.val}), ty.next)
    }

    pub fn parse_parameter(s: &str, i: uint, lim: uint) ->
        Parsed<Option<uint>> {
        if i >= lim { return Parsed::new(None, i); }

        match peek_num(s, i, lim) {
            Some(num) if num.next < lim && s[num.next] == '$' as u8 =>
                Parsed::new(Some(num.val), num.next + 1),
            _ => Parsed::new(None, i)
        }
    }

    pub fn parse_flags(s: &str, i: uint, lim: uint) -> Parsed<~[Flag]> {
        let mut i = i;
        let mut flags = ~[];

        while i < lim {
            let f = match s[i] as char {
                '-' => FlagLeftJustify,
                '0' => FlagLeftZeroPad,
                ' ' => FlagSpaceForSign,
                '+' => FlagSignAlways,
                '#' => FlagAlternate,
                _ => break
            };

            flags.push(f);
            i += 1;
        }

        Parsed::new(flags, i)
    }

    pub fn parse_count(s: &str, i: uint, lim: uint) -> Parsed<Count> {
        if i >= lim {
            Parsed::new(CountImplied, i)
        } else if s[i] == '*' as u8 {
            let param = parse_parameter(s, i + 1, lim);
            let j = param.next;

            match param.val {
                None => Parsed::new(CountIsNextParam, j),
                Some(n) => Parsed::new(CountIsParam(n), j)
            }
        } else {
            match peek_num(s, i, lim) {
                None => Parsed::new(CountImplied, i),
                Some(num) => Parsed::new(CountIs(num.val), num.next)
            }
        }
    }

    pub fn parse_precision(s: &str, i: uint, lim: uint) -> Parsed<Count> {
        if i < lim && s[i] == '.' as u8 {
            let count = parse_count(s, i + 1, lim);

            // If there were no digits specified, i.e. the precision
            // was ".", then the precision is 0
            match count.val {
                CountImplied => Parsed::new(CountIs(0), count.next),
                _ => count
            }
        } else {
            Parsed::new(CountImplied, i)
        }
    }

    pub fn parse_type(s: &str, i: uint, lim: uint, err: ErrorFn) ->
        Parsed<Ty> {
        if i >= lim { err("missing type in conversion"); }

        // FIXME (#2249): Do we really want two signed types here?
        // How important is it to be printf compatible?
        let t = match s[i] as char {
            'b' => TyBool,
            's' => TyStr,
            'c' => TyChar,
            'd' | 'i' => TyInt(Signed),
            'u' => TyInt(Unsigned),
            'x' => TyHex(CaseLower),
            'X' => TyHex(CaseUpper),
            't' => TyBits,
            'o' => TyOctal,
            'f' => TyFloat,
            '?' => TyPoly,
            _ => err(fmt!("unknown type in conversion: %c", s.char_at(i)))
        };

        Parsed::new(t, i + 1)
    }

    #[cfg(test)]
    fn die(s: &str) -> ! { fail!(s.to_owned()) }

    #[test]
    fn test_parse_count() {
        fn test(s: &str, count: Count, next: uint) -> bool {
            parse_count(s, 0, s.len()) == Parsed::new(count, next)
        }

        assert!(test("", CountImplied, 0));
        assert!(test("*", CountIsNextParam, 1));
        assert!(test("*1", CountIsNextParam, 1));
        assert!(test("*1$", CountIsParam(1), 3));
        assert!(test("123", CountIs(123), 3));
    }

    #[test]
    fn test_parse_flags() {
        fn pack(fs: &[Flag]) -> uint {
            fs.iter().fold(0, |p, &f| p | (1 << f as uint))
        }

        fn test(s: &str, flags: &[Flag], next: uint) {
            let f = parse_flags(s, 0, s.len());
            assert_eq!(pack(f.val), pack(flags));
            assert_eq!(f.next, next);
        }

        test("", [], 0);
        test("!#-+ 0", [], 0);
        test("#-+", [FlagAlternate, FlagLeftJustify, FlagSignAlways], 3);
        test(" 0", [FlagSpaceForSign, FlagLeftZeroPad], 2);
    }

    #[test]
    fn test_parse_fmt_string() {
        assert!(parse_fmt_string("foo %s bar", die) == ~[
            PieceString(~"foo "),
            PieceConv(Conv {
                param: None,
                flags: ~[],
                width: CountImplied,
                precision: CountImplied,
                ty: TyStr,
            }),
            PieceString(~" bar")]);

        assert!(parse_fmt_string("%s", die) == ~[
            PieceConv(Conv {
                param: None,
                flags: ~[],
                width: CountImplied,
                precision: CountImplied,
                ty: TyStr,
            })]);

        assert!(parse_fmt_string("%%%%", die) == ~[
            PieceString(~"%"), PieceString(~"%")]);
    }

    #[test]
    fn test_parse_parameter() {
        fn test(s: &str, param: Option<uint>, next: uint) -> bool {
            parse_parameter(s, 0, s.len()) == Parsed::new(param, next)
        }

        assert!(test("", None, 0));
        assert!(test("foo", None, 0));
        assert!(test("123", None, 0));
        assert!(test("123$", Some(123), 4));
    }

    #[test]
    fn test_parse_precision() {
        fn test(s: &str, count: Count, next: uint) -> bool {
            parse_precision(s, 0, s.len()) == Parsed::new(count, next)
        }

        assert!(test("", CountImplied, 0));
        assert!(test(".", CountIs(0), 1));
        assert!(test(".*", CountIsNextParam, 2));
        assert!(test(".*1", CountIsNextParam, 2));
        assert!(test(".*1$", CountIsParam(1), 4));
        assert!(test(".123", CountIs(123), 4));
    }

    #[test]
    fn test_parse_type() {
        fn test(s: &str, ty: Ty) -> bool {
            parse_type(s, 0, s.len(), die) == Parsed::new(ty, 1)
        }

        assert!(test("b", TyBool));
        assert!(test("c", TyChar));
        assert!(test("d", TyInt(Signed)));
        assert!(test("f", TyFloat));
        assert!(test("i", TyInt(Signed)));
        assert!(test("o", TyOctal));
        assert!(test("s", TyStr));
        assert!(test("t", TyBits));
        assert!(test("x", TyHex(CaseLower)));
        assert!(test("X", TyHex(CaseUpper)));
        assert!(test("?", TyPoly));
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_parse_type_missing() {
        parse_type("", 0, 0, die);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_parse_type_unknown() {
        parse_type("!", 0, 1, die);
    }

    #[test]
    fn test_peek_num() {
        let s1 = "";
        assert!(peek_num(s1, 0, s1.len()).is_none());

        let s2 = "foo";
        assert!(peek_num(s2, 0, s2.len()).is_none());

        let s3 = "123";
        assert_eq!(peek_num(s3, 0, s3.len()), Some(Parsed::new(123, 3)));

        let s4 = "123foo";
        assert_eq!(peek_num(s4, 0, s4.len()), Some(Parsed::new(123, 3)));
    }
}

// Functions used by the fmt extension at runtime. For now there are a lot of
// decisions made a runtime. If it proves worthwhile then some of these
// conditions can be evaluated at compile-time. For now though it's cleaner to
// implement it this way, I think.
#[doc(hidden)]
#[allow(non_uppercase_statics)]
pub mod rt {
    use float;
    use str;
    use sys;
    use num;
    use uint;
    use vec;
    use option::{Some, None, Option};

    pub static flag_none : u32 = 0u32;
    pub static flag_left_justify   : u32 = 0b00000000000001u32;
    pub static flag_left_zero_pad  : u32 = 0b00000000000010u32;
    pub static flag_space_for_sign : u32 = 0b00000000000100u32;
    pub static flag_sign_always    : u32 = 0b00000000001000u32;
    pub static flag_alternate      : u32 = 0b00000000010000u32;

    pub enum Count { CountIs(uint), CountImplied, }

    pub enum Ty { TyDefault, TyBits, TyHexUpper, TyHexLower, TyOctal, }

    pub struct Conv {
        flags: u32,
        width: Count,
        precision: Count,
        ty: Ty,
    }

    pub fn conv_int(cv: Conv, i: int, buf: &mut ~str) {
        let radix = 10;
        let prec = get_int_precision(cv);
        let s : ~str = uint_to_str_prec(num::abs(i) as uint, radix, prec);

        let head = if i >= 0 {
            if have_flag(cv.flags, flag_sign_always) {
                Some('+')
            } else if have_flag(cv.flags, flag_space_for_sign) {
                Some(' ')
            } else {
                None
            }
        } else { Some('-') };
        pad(cv, s, head, PadSigned, buf);
    }
    pub fn conv_uint(cv: Conv, u: uint, buf: &mut ~str) {
        let prec = get_int_precision(cv);
        let rs =
            match cv.ty {
              TyDefault => uint_to_str_prec(u, 10, prec),
              TyHexLower => uint_to_str_prec(u, 16, prec),

              // FIXME: #4318 Instead of to_ascii and to_str_ascii, could use
              // to_ascii_consume and to_str_consume to not do a unnecessary copy.
              TyHexUpper => {
                let s = uint_to_str_prec(u, 16, prec);
                s.to_ascii().to_upper().to_str_ascii()
              }
              TyBits => uint_to_str_prec(u, 2, prec),
              TyOctal => uint_to_str_prec(u, 8, prec)
            };
        pad(cv, rs, None, PadUnsigned, buf);
    }
    pub fn conv_bool(cv: Conv, b: bool, buf: &mut ~str) {
        let s = if b { "true" } else { "false" };
        // run the boolean conversion through the string conversion logic,
        // giving it the same rules for precision, etc.
        conv_str(cv, s, buf);
    }
    pub fn conv_char(cv: Conv, c: char, buf: &mut ~str) {
        pad(cv, "", Some(c), PadNozero, buf);
    }
    pub fn conv_str(cv: Conv, s: &str, buf: &mut ~str) {
        // For strings, precision is the maximum characters
        // displayed
        let unpadded = match cv.precision {
          CountImplied => s,
          CountIs(max) => if (max as uint) < s.char_len() {
            s.slice(0, max as uint)
          } else {
            s
          }
        };
        pad(cv, unpadded, None, PadNozero, buf);
    }
    pub fn conv_float(cv: Conv, f: float, buf: &mut ~str) {
        let (to_str, digits) = match cv.precision {
              CountIs(c) => (float::to_str_exact, c as uint),
              CountImplied => (float::to_str_digits, 6u)
        };
        let s = to_str(f, digits);
        let head = if 0.0 <= f {
            if have_flag(cv.flags, flag_sign_always) {
                Some('+')
            } else if have_flag(cv.flags, flag_space_for_sign) {
                Some(' ')
            } else {
                None
            }
        } else { None };
        pad(cv, s, head, PadFloat, buf);
    }
    pub fn conv_poly<T>(cv: Conv, v: &T, buf: &mut ~str) {
        let s = sys::log_str(v);
        conv_str(cv, s, buf);
    }

    // Convert a uint to string with a minimum number of digits.  If precision
    // is 0 and num is 0 then the result is the empty string. Could move this
    // to uint: but it doesn't seem all that useful.
    pub fn uint_to_str_prec(num: uint, radix: uint, prec: uint) -> ~str {
        return if prec == 0u && num == 0u {
                ~""
            } else {
                let s = uint::to_str_radix(num, radix);
                let len = s.char_len();
                if len < prec {
                    let diff = prec - len;
                    let pad = str::from_chars(vec::from_elem(diff, '0'));
                    pad + s
                } else { s }
            };
    }
    pub fn get_int_precision(cv: Conv) -> uint {
        return match cv.precision {
              CountIs(c) => c as uint,
              CountImplied => 1u
            };
    }

    #[deriving(Eq)]
    pub enum PadMode { PadSigned, PadUnsigned, PadNozero, PadFloat }

    pub fn pad(cv: Conv, s: &str, head: Option<char>, mode: PadMode,
               buf: &mut ~str) {
        let headsize = match head { Some(_) => 1, _ => 0 };
        let uwidth : uint = match cv.width {
            CountImplied => {
                for head.iter().advance |&c| {
                    buf.push_char(c);
                }
                return buf.push_str(s);
            }
            CountIs(width) => { width as uint }
        };
        let strlen = s.char_len() + headsize;
        if uwidth <= strlen {
            for head.iter().advance |&c| {
                buf.push_char(c);
            }
            return buf.push_str(s);
        }
        let mut padchar = ' ';
        let diff = uwidth - strlen;
        if have_flag(cv.flags, flag_left_justify) {
            for head.iter().advance |&c| {
                buf.push_char(c);
            }
            buf.push_str(s);
            for diff.times {
                buf.push_char(padchar);
            }
            return;
        }
        let (might_zero_pad, signed) = match mode {
          PadNozero   => (false, true),
          PadSigned   => (true, true),
          PadFloat    => (true, true),
          PadUnsigned => (true, false)
        };
        fn have_precision(cv: Conv) -> bool {
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

        if signed && zero_padding {
            for head.iter().advance |&head| {
                if head == '+' || head == '-' || head == ' ' {
                    buf.push_char(head);
                    buf.push_str(padstr);
                    buf.push_str(s);
                    return;
                }
            }
        }
        buf.push_str(padstr);
        for head.iter().advance |&c| {
            buf.push_char(c);
        }
        buf.push_str(s);
    }
    #[inline]
    pub fn have_flag(flags: u32, f: u32) -> bool {
        flags & f != 0
    }
}

// Bulk of the tests are in src/test/run-pass/syntax-extension-fmt.rs
#[cfg(test)]
mod test {
    #[test]
    fn fmt_slice() {
        let s = "abc";
        let _s = fmt!("%s", s);
    }
}
