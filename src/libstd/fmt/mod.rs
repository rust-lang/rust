// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

use cast;
use libc;
use str;
use uint;
use util;
use vec;

#[cfg(not(test))]
use super::{int, sys};

pub mod ct;

/// A struct to represent both where to emit formatting strings to and how they
/// should be formatted. A mutable version of this is passed to all formatting
/// traits.
pub struct Formatter<'self> {
    /// Flags for formatting (packed version of ct::Flag)
    flags: uint,
    /// Character used as 'fill' whenever there is alignment
    fill: char,
    /// Boolean indication of whether the output should be left-aligned
    alignleft: bool,
    /// Optionally specified integer width that the output should be
    width: Option<uint>,
    /// Optionally specified precision for numeric types
    precision: Option<uint>,

    /// Output buffer.
    // XXX: this should be '&mut io::Writer' when borrowed traits work better
    buf: &'self mut ~str,

    priv curarg: vec::VecIterator<'self, Argument<'self>>,
    priv args: &'self [Argument<'self>],
    priv named_args: &'self [(&'self str, Argument<'self>)],
}

/// This struct represents the generic "argument" which is taken by the Xprintf
/// family of functions. It contains a function to format the given value. At
/// compile time it is ensured that the function and the value have the correct
/// types, and then this struct is used to canonicalize arguments to one type.
pub struct Argument<'self> {
    priv formatter: extern "Rust" fn(&util::Void, &mut Formatter),
    priv value: &'self util::Void,
}

#[allow(missing_doc)] #[cfg(not(test))] #[fmt="b"]
pub trait Bool { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="c"]
pub trait Char { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="d"]
pub trait Signed { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="u"] #[fmt="i"]
pub trait Unsigned { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="o"]
pub trait Octal { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="t"]
pub trait Binary { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="x"]
pub trait LowerHex { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="X"]
pub trait UpperHex { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="s"]
pub trait String { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="?"]
pub trait Poly { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)] #[cfg(not(test))] #[fmt="p"]
pub trait Pointer { fn fmt(&Self, &mut Formatter); }

/// The sprintf function takes a format string, a list of arguments, a list of
/// named arguments, and it returns the resulting formatted string.
///
/// This is currently an unsafe function because the types of all arguments
/// aren't verified by immediate callers of this function. This currently does
/// not validate that the correct types of arguments are specified for each
/// format specifier, nor that each argument itself contains the right function
/// for formatting the right type value. Because of this, the function is marked
/// as `unsafe` if this is being called manually.
///
/// Thankfully the rust compiler provides the macro `ifmt!` which will perform
/// all of this validation at compile-time and provides a safe interface for
/// invoking this function.
///
/// # Arguments
///
///   * fmts - the format string to control output
///   * args - the list of arguments to the format string. These are only the
///            positional arguments (not named)
///   * named_args - this is a list of named arguments, the first value of the
///                  pair is the name, and the second is the actual argument.
///
/// Note that this function assumes that there are enough arguments and
/// named_args for the format string, and that each named arg in the format
/// string is also present in named_args.
pub unsafe fn sprintf(fmts: &str, args: &[Argument],
                      named_args: &[(&str, Argument)]) -> ~str {
    let mut output = ~"";
    {
        let mut fmt = Formatter {
            flags: 0,
            width: None,
            precision: None,
            buf: &mut output,
            alignleft: false,
            fill: ' ',
            curarg: args.iter(),
            args: args,
            named_args: named_args,
        };

        for ct::Parser::new(fmts, |_| libc::abort()).advance |piece| {
            fmt.run(&piece, None);
        }
    }
    return output;
}

impl<'self> Formatter<'self> {
    fn run(&mut self, piece: &ct::Piece<'self>, cur: Option<&str>) {
        let setcount = |slot: &mut Option<uint>, cnt: &ct::Count| {
            match *cnt {
                ct::CountIs(n) => { *slot = Some(n); }
                ct::CountImplied => { *slot = None; }
                ct::CountIsParam(i) => {
                    let v = self.args[i].value;
                    unsafe { *slot = Some(*(v as *util::Void as *uint)); }
                }
                ct::CountIsNextParam => {
                    let v = self.curarg.next().unwrap().value;
                    unsafe { *slot = Some(*(v as *util::Void as *uint)); }
                }
            }
        };

        match *piece {
            ct::String(s) => { self.buf.push_str(s); }
            ct::CurrentArgument => { self.buf.push_str(cur.unwrap()); }
            ct::Argument(ref arg) => {
                // Fill in the format parameters into the formatter
                match arg.format.fill {
                    Some(c) => { self.fill = c; }
                    None => { self.fill = ' '; }
                }
                match arg.format.align {
                    Some(ct::AlignRight) => { self.alignleft = false; }
                    Some(ct::AlignLeft) | None => { self.alignleft = true; }
                }
                self.flags = arg.format.flags;
                setcount(&mut self.width, &arg.format.width);
                setcount(&mut self.precision, &arg.format.precision);

                // Extract the correct argument
                let value = match arg.position {
                    ct::ArgumentNext => { *self.curarg.next().unwrap() }
                    ct::ArgumentIs(i) => self.args[i],
                    ct::ArgumentNamed(name) => {
                        match self.named_args.iter().find_(|& &(s, _)| s == name){
                            Some(&(_, arg)) => arg,
                            None => {
                                fail!(fmt!("No argument named `%s`", name))
                            }
                        }
                    }
                };

                // Then actually do some printing
                match arg.method {
                    None => { (value.formatter)(value.value, self); }
                    Some(ref method) => { self.execute(*method, value); }
                }
            }
        }
    }

    fn execute(&mut self, method: &ct::Method<'self>, arg: Argument) {
        match *method {
            // Pluralization is selection upon a numeric value specified as the
            // parameter.
            ct::Plural(offset, ref selectors, ref default) => {
                // This is validated at compile-time to be a pointer to a
                // '&uint' value.
                let value: &uint = unsafe { cast::transmute(arg.value) };
                let value = *value;

                // First, attempt to match against explicit values without the
                // offsetted value
                for selectors.iter().advance |s| {
                    match s.selector {
                        Right(val) if value == val => {
                            return self.runplural(value, s.result);
                        }
                        _ => {}
                    }
                }

                // Next, offset the value and attempt to match against the
                // keyword selectors.
                let value = value - offset.get_or_default(0);
                for selectors.iter().advance |s| {
                    let run = match s.selector {
                        Left(ct::Zero) => value == 0,
                        Left(ct::One) => value == 1,
                        Left(ct::Two) => value == 2,

                        // XXX: Few/Many should have a user-specified boundary
                        //      One possible option would be in the function
                        //      pointer of the 'arg: Argument' struct.
                        Left(ct::Few) => value < 8,
                        Left(ct::Many) => value >= 8,

                        Right(*) => false
                    };
                    if run {
                        return self.runplural(value, s.result);
                    }
                }

                self.runplural(value, *default);
            }

            // Select is just a matching against the string specified.
            ct::Select(ref selectors, ref default) => {
                // This is validated at compile-time to be a pointer to a
                // string slice,
                let value: & &str = unsafe { cast::transmute(arg.value) };
                let value = *value;

                for selectors.iter().advance |s| {
                    if s.selector == value {
                        for s.result.iter().advance |piece| {
                            self.run(piece, Some(value));
                        }
                        return;
                    }
                }
                for default.iter().advance |piece| {
                    self.run(piece, Some(value));
                }
            }
        }
    }

    fn runplural(&mut self, value: uint, pieces: &[ct::Piece<'self>]) {
        do uint::to_str_bytes(value, 10) |buf| {
            let valuestr = str::from_bytes_slice(buf);
            for pieces.iter().advance |piece| {
                self.run(piece, Some(valuestr));
            }
        }
    }
}

/// This is a function which calls are emitted to by the compiler itself to
/// create the Argument structures that are passed into the `sprintf` function.
#[doc(hidden)]
pub fn argument<'a, T>(f: extern "Rust" fn(&T, &mut Formatter),
                       t: &'a T) -> Argument<'a> {
    unsafe {
        Argument {
            formatter: cast::transmute(f),
            value: cast::transmute(t)
        }
    }
}

/// When the compiler determines that the type of an argument *must* be a string
/// (such as for select), then it invokes this method.
#[doc(hidden)]
pub fn argumentstr<'a>(s: &'a &str) -> Argument<'a> {
    argument(String::fmt, s)
}

/// When the compiler determines that the type of an argument *must* be a uint
/// (such as for plural), then it invokes this method.
#[doc(hidden)]
pub fn argumentuint<'a>(s: &'a uint) -> Argument<'a> {
    argument(Unsigned::fmt, s)
}

// Implementations of the core formatting traits

#[cfg(not(test))]
impl Bool for bool {
    fn fmt(b: &bool, f: &mut Formatter) {
        String::fmt(&(if *b {"true"} else {"false"}), f);
    }
}

#[cfg(not(test))]
impl<'self> String for &'self str {
    fn fmt(s: & &'self str, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.push_str(*s)
    }
}

#[cfg(not(test))]
impl Char for char {
    fn fmt(c: &char, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.push_char(*c);
    }
}

#[cfg(not(test))]
impl Signed for int {
    fn fmt(c: &int, f: &mut Formatter) {
        // XXX: formatting args
        do int::to_str_bytes(*c, 10) |buf| {
            f.buf.push_str(str::from_bytes_slice(buf));
        }
    }
}

#[cfg(not(test))]
impl Unsigned for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 10) |buf| {
            f.buf.push_str(str::from_bytes_slice(buf));
        }
    }
}

#[cfg(not(test))]
impl Octal for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 8) |buf| {
            f.buf.push_str(str::from_bytes_slice(buf));
        }
    }
}

#[cfg(not(test))]
impl LowerHex for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 16) |buf| {
            f.buf.push_str(str::from_bytes_slice(buf));
        }
    }
}

#[cfg(not(test))]
impl UpperHex for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 16) |buf| {
            let mut local = [0u8, ..16];
            for local.mut_iter().zip(buf.iter()).advance |(l, &b)| {
                *l = match b as char {
                    'a' .. 'f' => (b - 'a' as u8) + 'A' as u8,
                    _ => b,
                };
            }
            f.buf.push_str(str::from_bytes_slice(local.slice_to(buf.len())));
        }
    }
}

#[cfg(not(test))]
impl<T> Poly for T {
    fn fmt(t: &T, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.push_str(sys::log_str(t));
    }
}

// n.b. use 'const' to get an implementation for both '*mut' and '*' at the same
//      time.
#[cfg(not(test))]
impl<T> Pointer for *const T {
    fn fmt(t: &*const T, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.push_str("0x");
        LowerHex::fmt(&(*t as uint), f);
    }
}

#[cfg(test)]
mod tests {
    use realstd::fmt;

    use c = realstd::fmt::argument;

    fn run(s: &str, args: &[fmt::Argument]) -> ~str {
        unsafe { fmt::sprintf(s, args, []) }
    }

    #[test]
    fn simple() {
        assert_eq!(run("hello", []), ~"hello");
        assert_eq!(run("", []), ~"");
        assert_eq!(run("\\#", []), ~"#");
        assert_eq!(run("a\\#b", []), ~"a#b");
    }

    #[test]
    fn simple_args() {
        let a: uint = 1;
        let b: int = 2;
        let s = "hello";
        assert_eq!(run("a{}b", [c(fmt::Unsigned::fmt, &a)]), ~"a1b");
        assert_eq!(run("{} yay", [c(fmt::Signed::fmt, &b)]), ~"2 yay");
        assert_eq!(run("{}", [c(fmt::String::fmt, &s)]), ~"hello");
        assert_eq!(run("{} {2} {} {1} {}",
                       [c(fmt::String::fmt, &s),
                        c(fmt::Unsigned::fmt, &a),
                        c(fmt::Signed::fmt, &b)]),
                   ~"hello 2 1 1 2");
    }

    #[test]
    fn plural_method() {
        let one = 1u;
        let one = c(fmt::Unsigned::fmt, &one);
        let two = 2u;
        let two = c(fmt::Unsigned::fmt, &two);
        let three = 3u;
        let three = c(fmt::Unsigned::fmt, &three);
        let zero = 0u;
        let zero = c(fmt::Unsigned::fmt, &zero);
        let six = 6u;
        let six = c(fmt::Unsigned::fmt, &six);
        let ten = 10u;
        let ten = c(fmt::Unsigned::fmt, &ten);

        assert_eq!(run("{0,plural, other{a}}", [one]), ~"a");
        assert_eq!(run("{0,plural, =1{b} other{a}}", [one]), ~"b");
        assert_eq!(run("{0,plural, =0{b} other{a}}", [one]), ~"a");
        assert_eq!(run("{0,plural, =1{a} one{b} other{c}}", [one]), ~"a");
        assert_eq!(run("{0,plural, one{b} other{c}}", [one]), ~"b");
        assert_eq!(run("{0,plural, two{b} other{c}}", [two]), ~"b");
        assert_eq!(run("{0,plural, zero{b} other{c}}", [zero]), ~"b");
        assert_eq!(run("{0,plural, offset:1 zero{b} other{c}}", [zero]), ~"c");
        assert_eq!(run("{0,plural, offset:1 zero{b} other{c}}", [one]), ~"b");
        assert_eq!(run("{0,plural, few{a} many{b} other{c}}", [three]), ~"a");
        assert_eq!(run("{0,plural, few{a} many{b} other{c}}", [six]), ~"a");
        assert_eq!(run("{0,plural, few{a} many{b} other{c}}", [ten]), ~"b");
        assert_eq!(run("{0,plural, few{a} other{c}}", [ten]), ~"c");
        assert_eq!(run("{0,plural, few{a} other{#}}", [ten]), ~"10");
    }

    #[test]
    fn select_method() {
        let a = "a";
        let b = "b";
        let a = c(fmt::String::fmt, &a);
        let b = c(fmt::String::fmt, &b);

        assert_eq!(run("{0,select, other{a}}", [a]), ~"a");
        assert_eq!(run("{0,select, a{a} other{b}}", [a]), ~"a");
        assert_eq!(run("{0,select, a{#} other{b}}", [a]), ~"a");
        assert_eq!(run("{0,select, a{a} other{b}}", [b]), ~"b");
        assert_eq!(run("{0,select, a{s} b{t} other{b}}", [b]), ~"t");
        assert_eq!(run("{0,select, a{s} b{t} other{b}}", [a]), ~"s");
    }

    #[test]
    fn nested_method() {
        let a = "a";
        let a = c(fmt::String::fmt, &a);

        assert_eq!(run("{0,select,other{{0,select,a{a} other{b}}}}", [a]), ~"a");
        assert_eq!(run("{0,select,other{{0,select,c{a} other{b}}}}", [a]), ~"b");
        assert_eq!(run("{0,select,other{#{0,select,c{a} other{b}}}}", [a]), ~"ab");
    }
}
