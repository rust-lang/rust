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
use int;
use rt::io::Decorator;
use rt::io::mem::MemWriter;
use rt::io;
use str;
use sys;
use uint;
use util;
use vec;

pub mod parse;
pub mod rt;

/// A struct to represent both where to emit formatting strings to and how they
/// should be formatted. A mutable version of this is passed to all formatting
/// traits.
pub struct Formatter<'self> {
    /// Flags for formatting (packed version of rt::Flag)
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
    buf: &'self mut io::Writer,

    priv curarg: vec::VecIterator<'self, Argument<'self>>,
    priv args: &'self [Argument<'self>],
}

/// This struct represents the generic "argument" which is taken by the Xprintf
/// family of functions. It contains a function to format the given value. At
/// compile time it is ensured that the function and the value have the correct
/// types, and then this struct is used to canonicalize arguments to one type.
pub struct Argument<'self> {
    priv formatter: extern "Rust" fn(&util::Void, &mut Formatter),
    priv value: &'self util::Void,
}

#[allow(missing_doc)]
pub trait Bool { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Char { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Signed { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Unsigned { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Octal { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Binary { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait LowerHex { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait UpperHex { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait String { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Poly { fn fmt(&Self, &mut Formatter); }
#[allow(missing_doc)]
pub trait Pointer { fn fmt(&Self, &mut Formatter); }

/// The sprintf function takes a precompiled format string and a list of
/// arguments, to return the resulting formatted string.
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
///   * fmts - the precompiled format string to emit.
///   * args - the list of arguments to the format string. These are only the
///            positional arguments (not named)
///
/// Note that this function assumes that there are enough arguments for the
/// format string.
pub unsafe fn sprintf(fmt: &[rt::Piece], args: &[Argument]) -> ~str {
    let output = MemWriter::new();
    {
        let mut formatter = Formatter {
            flags: 0,
            width: None,
            precision: None,
            // FIXME(#8248): shouldn't need a transmute
            buf: cast::transmute(&output as &io::Writer),
            alignleft: false,
            fill: ' ',
            args: args,
            curarg: args.iter(),
        };
        for piece in fmt.iter() {
            formatter.run(piece, None);
        }
    }
    return str::from_bytes_owned(output.inner());
}

impl<'self> Formatter<'self> {
    fn run(&mut self, piece: &rt::Piece, cur: Option<&str>) {
        let setcount = |slot: &mut Option<uint>, cnt: &parse::Count| {
            match *cnt {
                parse::CountIs(n) => { *slot = Some(n); }
                parse::CountImplied => { *slot = None; }
                parse::CountIsParam(i) => {
                    let v = self.args[i].value;
                    unsafe { *slot = Some(*(v as *util::Void as *uint)); }
                }
                parse::CountIsNextParam => {
                    let v = self.curarg.next().unwrap().value;
                    unsafe { *slot = Some(*(v as *util::Void as *uint)); }
                }
            }
        };

        match *piece {
            rt::String(s) => { self.buf.write(s.as_bytes()); }
            rt::CurrentArgument(()) => { self.buf.write(cur.unwrap().as_bytes()); }
            rt::Argument(ref arg) => {
                // Fill in the format parameters into the formatter
                self.fill = arg.format.fill;
                self.alignleft = arg.format.alignleft;
                self.flags = arg.format.flags;
                setcount(&mut self.width, &arg.format.width);
                setcount(&mut self.precision, &arg.format.precision);

                // Extract the correct argument
                let value = match arg.position {
                    rt::ArgumentNext => { *self.curarg.next().unwrap() }
                    rt::ArgumentIs(i) => self.args[i],
                };

                // Then actually do some printing
                match arg.method {
                    None => { (value.formatter)(value.value, self); }
                    Some(ref method) => { self.execute(*method, value); }
                }
            }
        }
    }

    fn execute(&mut self, method: &rt::Method, arg: Argument) {
        match *method {
            // Pluralization is selection upon a numeric value specified as the
            // parameter.
            rt::Plural(offset, ref selectors, ref default) => {
                // This is validated at compile-time to be a pointer to a
                // '&uint' value.
                let value: &uint = unsafe { cast::transmute(arg.value) };
                let value = *value;

                // First, attempt to match against explicit values without the
                // offsetted value
                for s in selectors.iter() {
                    match s.selector {
                        Right(val) if value == val => {
                            return self.runplural(value, s.result);
                        }
                        _ => {}
                    }
                }

                // Next, offset the value and attempt to match against the
                // keyword selectors.
                let value = value - match offset { Some(i) => i, None => 0 };
                for s in selectors.iter() {
                    let run = match s.selector {
                        Left(parse::Zero) => value == 0,
                        Left(parse::One) => value == 1,
                        Left(parse::Two) => value == 2,

                        // XXX: Few/Many should have a user-specified boundary
                        //      One possible option would be in the function
                        //      pointer of the 'arg: Argument' struct.
                        Left(parse::Few) => value < 8,
                        Left(parse::Many) => value >= 8,

                        Right(*) => false
                    };
                    if run {
                        return self.runplural(value, s.result);
                    }
                }

                self.runplural(value, *default);
            }

            // Select is just a matching against the string specified.
            rt::Select(ref selectors, ref default) => {
                // This is validated at compile-time to be a pointer to a
                // string slice,
                let value: & &str = unsafe { cast::transmute(arg.value) };
                let value = *value;

                for s in selectors.iter() {
                    if s.selector == value {
                        for piece in s.result.iter() {
                            self.run(piece, Some(value));
                        }
                        return;
                    }
                }
                for piece in default.iter() {
                    self.run(piece, Some(value));
                }
            }
        }
    }

    fn runplural(&mut self, value: uint, pieces: &[rt::Piece]) {
        do uint::to_str_bytes(value, 10) |buf| {
            let valuestr = str::from_bytes_slice(buf);
            for piece in pieces.iter() {
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

impl Bool for bool {
    fn fmt(b: &bool, f: &mut Formatter) {
        String::fmt(&(if *b {"true"} else {"false"}), f);
    }
}

impl<'self> String for &'self str {
    fn fmt(s: & &'self str, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.write(s.as_bytes())
    }
}

impl Char for char {
    fn fmt(c: &char, f: &mut Formatter) {
        // XXX: formatting args
        // XXX: shouldn't require an allocation
        let mut s = ~"";
        s.push_char(*c);
        f.buf.write(s.as_bytes());
    }
}

impl Signed for int {
    fn fmt(c: &int, f: &mut Formatter) {
        // XXX: formatting args
        do int::to_str_bytes(*c, 10) |buf| {
            f.buf.write(buf);
        }
    }
}

impl Unsigned for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 10) |buf| {
            f.buf.write(buf);
        }
    }
}

impl Octal for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 8) |buf| {
            f.buf.write(buf);
        }
    }
}

impl LowerHex for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 16) |buf| {
            f.buf.write(buf);
        }
    }
}

impl UpperHex for uint {
    fn fmt(c: &uint, f: &mut Formatter) {
        // XXX: formatting args
        do uint::to_str_bytes(*c, 16) |buf| {
            let mut local = [0u8, ..16];
            for (l, &b) in local.mut_iter().zip(buf.iter()) {
                *l = match b as char {
                    'a' .. 'f' => (b - 'a' as u8) + 'A' as u8,
                    _ => b,
                };
            }
            f.buf.write(local.slice_to(buf.len()));
        }
    }
}

impl<T> Poly for T {
    fn fmt(t: &T, f: &mut Formatter) {
        // XXX: formatting args
        let s = sys::log_str(t);
        f.buf.write(s.as_bytes());
    }
}

// n.b. use 'const' to get an implementation for both '*mut' and '*' at the same
//      time.
impl<T> Pointer for *const T {
    fn fmt(t: &*const T, f: &mut Formatter) {
        // XXX: formatting args
        f.buf.write("0x".as_bytes());
        LowerHex::fmt(&(*t as uint), f);
    }
}

// If you expected tests to be here, look instead at the run-pass/ifmt.rs test,
// it's a lot easier than creating all of the rt::Piece structures here.
