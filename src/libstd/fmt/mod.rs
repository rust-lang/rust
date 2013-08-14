// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**!

# The Formatting Module

This module contains the runtime support for the `ifmt!` syntax extension. This
macro is implemented in the compiler to emit calls to this module in order to
format arguments at runtime into strings and streams.

The functions contained in this module should not normally be used in everyday
use cases of `ifmt!`. The assumptions made by these functions are unsafe for all
inputs, and the compiler performs a large amount of validation on the arguments
to `ifmt!` in order to ensure safety at runtime. While it is possible to call
these functions directly, it is not recommended to do so in the general case.

## Usage

The `ifmt!` macro is intended to be familiar to those coming from C's
printf/sprintf functions or Python's `str.format` function. In its current
revision, the `ifmt!` macro returns a `~str` type which is the result of the
formatting. In the future it will also be able to pass in a stream to format
arguments directly while performing minimal allocations.

Some examples of the `ifmt!` extension are:

~~~{.rust}
ifmt!("Hello")                  // => ~"Hello"
ifmt!("Hello, {:s}!", "world")  // => ~"Hello, world!"
ifmt!("The number is {:d}", 1)  // => ~"The number is 1"
ifmt!("{}", ~[3, 4])            // => ~"~[3, 4]"
ifmt!("{value}", value=4)       // => ~"4"
ifmt!("{} {}", 1, 2)            // => ~"1 2"
~~~

From these, you can see that the first argument is a format string. It is
required by the compiler for this to be a string literal; it cannot be a
variable passed in (in order to perform validity checking). The compiler will
then parse the format string and determine if the list of arguments provided is
suitable to pass to this format string.

### Positional parameters

Each formatting argument is allowed to specify which value argument it's
referencing, and if omitted it is assumed to be "the next argument". For
example, the format string `{} {} {}` would take three parameters, and they
would be formatted in the same order as they're given. The format string
`{2} {1} {0}`, however, would format arguments in reverse order.

A format string is required to use all of its arguments, otherwise it is a
compile-time error. You may refer to the same argument more than once in the
format string, although it must always be referred to with the same type.

### Named parameters

Rust itself does not have a Python-like equivalent of named parameters to a
function, but the `ifmt!` macro is a syntax extension which allows it to
leverage named parameters. Named parameters are listed at the end of the
argument list and have the syntax:

~~~
identifier '=' expression
~~~

It is illegal to put positional parameters (those without names) after arguments
which have names. Like positional parameters, it is illegal to provided named
parameters that are unused by the format string.

### Argument types

Each argument's type is dictated by the format string. It is a requirement that
every argument is only ever referred to by one type. When specifying the format
of an argument, however, a string like `{}` indicates no type. This is allowed,
and if all references to one argument do not provide a type, then the format `?`
is used (the type's rust-representation is printed). For example, this is an
invalid format string:

~~~
{0:d} {0:s}
~~~

Because the first argument is both referred to as an integer as well as a
string.

Because formatting is done via traits, there is no requirement that the
`d` format actually takes an `int`, but rather it simply requires a type which
ascribes to the `Signed` formatting trait. There are various parameters which do
require a particular type, however. Namely if the syntax `{:.*s}` is used, then
the number of characters to print from the string precedes the actual string and
must have the type `uint`. Although a `uint` can be printed with `{:u}`, it is
illegal to reference an argument as such. For example, this is another invalid
format string:

~~~
{:.*s} {0:u}
~~~

### Formatting traits

When requesting that an argument be formatted with a particular type, you are
actually requesting that an argument ascribes to a particular trait. This allows
multiple actual types to be formatted via `{:d}` (like `i8` as well as `int`).
The current mapping of types to traits is:

* `?` => Poly
* `d` => Signed
* `i` => Signed
* `u` => Unsigned
* `b` => Bool
* `c` => Char
* `o` => Octal
* `x` => LowerHex
* `X` => UpperHex
* `s` => String
* `p` => Pointer
* `t` => Binary
* `f` => Float

What this means is that any type of argument which implements the
`std::fmt::Binary` trait can then be formatted with `{:t}`. Implementations are
provided for these traits for a number of primitive types by the standard
library as well. Again, the default formatting type (if no other is specified)
is `?` which is defined for all types by default.

When implementing a format trait for your own time, you will have to implement a
method of the signature:

~~~
fn fmt(value: &T, f: &mut std::fmt::Formatter);
~~~

Your type will be passed by-reference in `value`, and then the function should
emit output into the `f.buf` stream. It is up to each format trait
implementation to correctly adhere to the requested formatting parameters. The
values of these parameters will be listed in the fields of the `Formatter`
struct. In order to help with this, the `Formatter` struct also provides some
helper methods.

## Internationalization

The formatting syntax supported by the `ifmt!` extension supports
internationalization by providing "methods" which execute various differnet
outputs depending on the input. The syntax and methods provided are similar to
other internationalization systems, so again nothing should seem alien.
Currently two methods are supported by this extension: "select" and "plural".

Each method will execute one of a number of clauses, and then the value of the
clause will become what's the result of the argument's format. Inside of the
cases, nested argument strings may be provided, but all formatting arguments
must not be done through implicit positional means. All arguments inside of each
case of a method must be explicitly selected by their name or their integer
position.

Furthermore, whenever a case is running, the special character `#` can be used
to reference the string value of the argument which was selected upon. As an
example:

~~~
ifmt!("{0, select, other{#}}", "hello") // => ~"hello"
~~~

This example is the equivalent of `{0:s}` essentially.

### Select

The select method is a switch over a `&str` parameter, and the parameter *must*
be of the type `&str`. An example of the syntax is:

~~~
{0, select, male{...} female{...} other{...}}
~~~

Breaking this down, the `0`-th argument is selected upon with the `select`
method, and then a number of cases follow. Each case is preceded by an
identifier which is the match-clause to execute the given arm. In this case,
there are two explicit cases, `male` and `female`. The case will be executed if
the string argument provided is an exact match to the case selected.

The `other` case is also a required case for all `select` methods. This arm will
be executed if none of the other arms matched the word being selected over.

### Plural

The plural method is a switch statement over a `uint` parameter, and the
parameter *must* be a `uint`. A plural method in its full glory can be specified
as:

~~~
{0, plural, offset=1 =1{...} two{...} many{...} other{...}}
~~~

To break this down, the first `0` indicates that this method is selecting over
the value of the first positional parameter to the format string. Next, the
`plural` method is being executed. An optionally-supplied `offset` is then given
which indicates a number to subtract from argument `0` when matching. This is
then followed by a list of cases.

Each case is allowed to supply a specific value to match upon with the syntax
`=N`. This case is executed if the value at argument `0` matches N exactly,
without taking the offset into account. A case may also be specified by one of
five keywords: `zero`, `one`, `two`, `few`, and `many`. These cases are matched
on after argument `0` has the offset taken into account. Currently the
definitions of `many` and `few` are hardcoded, but they are in theory defined by
the current locale.

Finally, all `plural` methods must have an `other` case supplied which will be
executed if none of the other cases match.

## Syntax

The syntax for the formatting language used is drawn from other languages, so it
should not be too alien. Arguments are formatted with python-like syntax,
meaning that arguments are surrounded by `{}` instead of the C-like `%`. The
actual grammar for the formatting syntax is:

~~~
format_string := <text> [ format <text> ] *
format := '{' [ argument ] [ ':' format_spec ] [ ',' function_spec ] '}'
argument := integer | identifier

format_spec := [[fill]align][sign]['#'][0][width]['.' precision][type]
fill := character
align := '<' | '>'
sign := '+' | '-'
width := count
precision := count | '*'
type := identifier | ''
count := parameter | integer
parameter := integer '$'

function_spec := plural | select
select := 'select' ',' ( identifier arm ) *
plural := 'plural' ',' [ 'offset:' integer ] ( selector arm ) *
selector := '=' integer | keyword
keyword := 'zero' | 'one' | 'two' | 'few' | 'many' | 'other'
arm := '{' format_string '}'
~~~

## Formatting Parameters

Each argument being formatted can be transformed by a number of formatting
parameters (corresponding to `format_spec` in the syntax above). These
parameters affect the string representation of what's being formatted. This
syntax draws heavily from Python's, so it may seem a bit familiar.

### Fill/Alignment

The fill character is provided normally in conjunction with the `width`
parameter. This indicates that if the value being formatted is smaller than
`width` some extra characters will be printed around it. The extra characters
are specified by `fill`, and the alignment can be one of two options:

* `<` - the argument is left-aligned in `width` columns
* `>` - the argument is right-aligned in `width` columns

### Sign/#/0

These can all be interpreted as flags for a particular formatter.

* '+' - This is intended for numeric types and indicates that the sign should
        always be printed. Positive signs are never printed by default, and the
        negative sign is only printed by default for the `Signed` trait. This
        flag indicates that the correct sign (+ or -) should always be printed.
* '-' - Currently not used
* '#' - This flag is indicates that the "alternate" form of printing should be
        used. By default, this only applies to the integer formatting traits and
        performs like:
    * `x` - precedes the argument with a "0x"
    * `X` - precedes the argument with a "0x"
    * `t` - precedes the argument with a "0b"
    * `o` - precedes the argument with a "0o"
* '0' - This is used to indicate for integer formats that the padding should
        both be done with a `0` character as well as be sign-aware. A format
        like `{:08d}` would yield `00000001` for the integer `1`, while the same
        format would yield `-0000001` for the integer `-1`. Notice that the
        negative version has one fewer zero than the positive version.

### Width

This is a parameter for the "minimum width" that the format should take up. If
the value's string does not fill up this many characters, then the padding
specified by fill/alignment will be used to take up the required space.

The default fill/alignment for non-numerics is a space and left-aligned. The
defaults for numeric formatters is also a space but with right-alignment. If the
'0' flag is specified for numerics, then the implicit fill character is '0'.

The value for the width can also be provided as a `uint` in the list of
parameters by using the `2$` syntax indicating that the second argument is a
`uint` specifying the width.

### Precision

For non-numeric types, this can be considered a "maximum width". If the
resulting string is longer than this width, then it is truncated down to this
many characters and only those are emitted.

For integral types, this has no meaning currently.

For floating-point types, this indicates how many digits after the decimal point
should be printed.

*/

use prelude::*;

use cast;
use char::Char;
use rt::io::Decorator;
use rt::io::mem::MemWriter;
use rt::io;
use str;
use sys;
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
    align: parse::Alignment,
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
#[allow(missing_doc)]
pub trait Float { fn fmt(&Self, &mut Formatter); }

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
            align: parse::AlignUnknown,
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

    // First up is the collection of functions used to execute a format string
    // at runtime. This consumes all of the compile-time statics generated by
    // the ifmt! syntax extension.

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
                self.align = arg.format.align;
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
        do ::uint::to_str_bytes(value, 10) |buf| {
            let valuestr = str::from_bytes_slice(buf);
            for piece in pieces.iter() {
                self.run(piece, Some(valuestr));
            }
        }
    }

    // Helper methods used for padding and processing formatting arguments that
    // all formatting traits can use.

    /// Performs the correct padding for an integer which has already been
    /// emitted into a byte-array. The byte-array should *not* contain the sign
    /// for the integer, that will be added by this method.
    ///
    /// # Arguments
    ///
    ///     * s - the byte array that the number has been formatted into
    ///     * alternate_prefix - if the '#' character (FlagAlternate) is
    ///       provided, this is the prefix to put in front of the number.
    ///       Currently this is 0x/0o/0b/etc.
    ///     * positive - whether the original integer was positive or not.
    ///
    /// This function will correctly account for the flags provided as well as
    /// the minimum width. It will not take precision into account.
    pub fn pad_integral(&mut self, s: &[u8], alternate_prefix: &str,
                        positive: bool) {
        use fmt::parse::{FlagAlternate, FlagSignPlus, FlagSignAwareZeroPad};

        let mut actual_len = s.len();
        if self.flags & 1 << (FlagAlternate as uint) != 0 {
            actual_len += alternate_prefix.len();
        }
        if self.flags & 1 << (FlagSignPlus as uint) != 0 {
            actual_len += 1;
        } else if !positive {
            actual_len += 1;
        }

        let mut signprinted = false;
        let sign = |this: &mut Formatter| {
            if !signprinted {
                if this.flags & 1 << (FlagSignPlus as uint) != 0 && positive {
                    this.buf.write(['+' as u8]);
                } else if !positive {
                    this.buf.write(['-' as u8]);
                }
                if this.flags & 1 << (FlagAlternate as uint) != 0 {
                    this.buf.write(alternate_prefix.as_bytes());
                }
                signprinted = true;
            }
        };

        let emit = |this: &mut Formatter| {
            sign(this);
            this.buf.write(s);
        };

        match self.width {
            None => { emit(self) }
            Some(min) if actual_len >= min => { emit(self) }
            Some(min) => {
                if self.flags & 1 << (FlagSignAwareZeroPad as uint) != 0 {
                    self.fill = '0';
                    sign(self);
                }
                do self.with_padding(min - actual_len, parse::AlignRight) |me| {
                    emit(me);
                }
            }
        }
    }

    /// This function takes a string slice and emits it to the internal buffer
    /// after applying the relevant formatting flags specified. The flags
    /// recognized for generic strings are:
    ///
    /// * width - the minimum width of what to emit
    /// * fill/align - what to emit and where to emit it if the string
    ///                provided needs to be padded
    /// * precision - the maximum length to emit, the string is truncated if it
    ///               is longer than this length
    ///
    /// Notably this function ignored the `flag` parameters
    pub fn pad(&mut self, s: &str) {
        // Make sure there's a fast path up front
        if self.width.is_none() && self.precision.is_none() {
            self.buf.write(s.as_bytes());
            return
        }
        // The `precision` field can be interpreted as a `max-width` for the
        // string being formatted
        match self.precision {
            Some(max) => {
                // If there's a maximum width and our string is longer than
                // that, then we must always have truncation. This is the only
                // case where the maximum length will matter.
                let char_len = s.char_len();
                if char_len >= max {
                    let nchars = ::uint::min(max, char_len);
                    self.buf.write(s.slice_chars(0, nchars).as_bytes());
                    return
                }
            }
            None => {}
        }

        // The `width` field is more of a `min-width` parameter at this point.
        match self.width {
            // If we're under the maximum length, and there's no minimum length
            // requirements, then we can just emit the string
            None => { self.buf.write(s.as_bytes()) }

            // If we're under the maximum width, check if we're over the minimum
            // width, if so it's as easy as just emitting the string.
            Some(width) if s.char_len() >= width => {
                self.buf.write(s.as_bytes())
            }

            // If we're under both the maximum and the minimum width, then fill
            // up the minimum width with the specified string + some alignment.
            Some(width) => {
                do self.with_padding(width - s.len(), parse::AlignLeft) |me| {
                    me.buf.write(s.as_bytes());
                }
            }
        }
    }

    fn with_padding(&mut self, padding: uint,
                    default: parse::Alignment, f: &fn(&mut Formatter)) {
        let align = match self.align {
            parse::AlignUnknown => default,
            parse::AlignLeft | parse::AlignRight => self.align
        };
        if align == parse::AlignLeft {
            f(self);
        }
        let mut fill = [0u8, ..4];
        let len = self.fill.encode_utf8(fill);
        for _ in range(0, padding) {
            self.buf.write(fill.slice_to(len));
        }
        if align == parse::AlignRight {
            f(self);
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
        f.pad(*s);
    }
}

impl Char for char {
    fn fmt(c: &char, f: &mut Formatter) {
        let mut utf8 = [0u8, ..4];
        let amt = c.encode_utf8(utf8);
        let s: &str = unsafe { cast::transmute(utf8.slice_to(amt)) };
        String::fmt(&s, f);
    }
}

macro_rules! int_base(($ty:ident, $into:ident, $base:expr,
                       $name:ident, $prefix:expr) => {
    impl $name for $ty {
        fn fmt(c: &$ty, f: &mut Formatter) {
            do ::$into::to_str_bytes(*c as $into, $base) |buf| {
                f.pad_integral(buf, $prefix, true);
            }
        }
    }
})
macro_rules! upper_hex(($ty:ident, $into:ident) => {
    impl UpperHex for $ty {
        fn fmt(c: &$ty, f: &mut Formatter) {
            do ::$into::to_str_bytes(*c as $into, 16) |buf| {
                upperhex(buf, f);
            }
        }
    }
})
// Not sure why, but this causes an "unresolved enum variant, struct or const"
// when inlined into the above macro...
#[doc(hidden)]
pub fn upperhex(buf: &[u8], f: &mut Formatter) {
    let mut local = [0u8, ..16];
    for i in ::iterator::range(0, buf.len()) {
        local[i] = match buf[i] as char {
            'a' .. 'f' => (buf[i] - 'a' as u8) + 'A' as u8,
            c => c as u8,
        }
    }
    f.pad_integral(local.slice_to(buf.len()), "0x", true);
}

// FIXME(#4375) shouldn't need an inner module
macro_rules! integer(($signed:ident, $unsigned:ident) => {
    mod $signed {
        use super::*;

        // Signed is special because it actuall emits the negative sign,
        // nothing else should do that, however.
        impl Signed for $signed {
            fn fmt(c: &$signed, f: &mut Formatter) {
                do ::$unsigned::to_str_bytes(c.abs() as $unsigned, 10) |buf| {
                    f.pad_integral(buf, "", *c >= 0);
                }
            }
        }
        int_base!($signed, $unsigned, 2, Binary, "0b")
        int_base!($signed, $unsigned, 8, Octal, "0o")
        int_base!($signed, $unsigned, 16, LowerHex, "0x")
        upper_hex!($signed, $unsigned)

        int_base!($unsigned, $unsigned, 2, Binary, "0b")
        int_base!($unsigned, $unsigned, 8, Octal, "0o")
        int_base!($unsigned, $unsigned, 10, Unsigned, "")
        int_base!($unsigned, $unsigned, 16, LowerHex, "0x")
        upper_hex!($unsigned, $unsigned)
    }
})

integer!(int, uint)
integer!(i8, u8)
integer!(i16, u16)
integer!(i32, u32)
integer!(i64, u64)

macro_rules! floating(($ty:ident) => {
    impl Float for $ty {
        fn fmt(f: &$ty, fmt: &mut Formatter) {
            // XXX: this shouldn't perform an allocation
            let s = match fmt.precision {
                Some(i) => ::$ty::to_str_exact(f.abs(), i),
                None => ::$ty::to_str_digits(f.abs(), 6)
            };
            fmt.pad_integral(s.as_bytes(), "", *f >= 0.0);
        }
    }
})
floating!(float)
floating!(f32)
floating!(f64)

impl<T> Poly for T {
    fn fmt(t: &T, f: &mut Formatter) {
        match (f.width, f.precision) {
            (None, None) => {
                // XXX: sys::log_str should have a variant which takes a stream
                //      and we should directly call that (avoids unnecessary
                //      allocations)
                let s = sys::log_str(t);
                f.buf.write(s.as_bytes());
            }

            // If we have a specified width for formatting, then we have to make
            // this allocation of a new string
            _ => {
                let s = sys::log_str(t);
                f.pad(s);
            }
        }
    }
}

// n.b. use 'const' to get an implementation for both '*mut' and '*' at the same
//      time.
impl<T> Pointer for *const T {
    fn fmt(t: &*const T, f: &mut Formatter) {
        f.flags |= 1 << (parse::FlagAlternate as uint);
        do ::uint::to_str_bytes(*t as uint, 16) |buf| {
            f.pad_integral(buf, "0x", true);
        }
    }
}

// If you expected tests to be here, look instead at the run-pass/ifmt.rs test,
// it's a lot easier than creating all of the rt::Piece structures here.
