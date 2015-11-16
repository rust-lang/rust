// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for formatting and printing strings
//!
//! This module contains the runtime support for the `format!` syntax extension.
//! This macro is implemented in the compiler to emit calls to this module in
//! order to format arguments at runtime into strings and streams.
//!
//! # Usage
//!
//! The `format!` macro is intended to be familiar to those coming from C's
//! printf/fprintf functions or Python's `str.format` function. In its current
//! revision, the `format!` macro returns a `String` type which is the result of
//! the formatting. In the future it will also be able to pass in a stream to
//! format arguments directly while performing minimal allocations.
//!
//! Some examples of the `format!` extension are:
//!
//! ```
//! format!("Hello");                 // => "Hello"
//! format!("Hello, {}!", "world");   // => "Hello, world!"
//! format!("The number is {}", 1);   // => "The number is 1"
//! format!("{:?}", (3, 4));          // => "(3, 4)"
//! format!("{value}", value=4);      // => "4"
//! format!("{} {}", 1, 2);           // => "1 2"
//! ```
//!
//! From these, you can see that the first argument is a format string. It is
//! required by the compiler for this to be a string literal; it cannot be a
//! variable passed in (in order to perform validity checking). The compiler
//! will then parse the format string and determine if the list of arguments
//! provided is suitable to pass to this format string.
//!
//! ## Positional parameters
//!
//! Each formatting argument is allowed to specify which value argument it's
//! referencing, and if omitted it is assumed to be "the next argument". For
//! example, the format string `{} {} {}` would take three parameters, and they
//! would be formatted in the same order as they're given. The format string
//! `{2} {1} {0}`, however, would format arguments in reverse order.
//!
//! Things can get a little tricky once you start intermingling the two types of
//! positional specifiers. The "next argument" specifier can be thought of as an
//! iterator over the argument. Each time a "next argument" specifier is seen,
//! the iterator advances. This leads to behavior like this:
//!
//! ```
//! format!("{1} {} {0} {}", 1, 2); // => "2 1 1 2"
//! ```
//!
//! The internal iterator over the argument has not been advanced by the time
//! the first `{}` is seen, so it prints the first argument. Then upon reaching
//! the second `{}`, the iterator has advanced forward to the second argument.
//! Essentially, parameters which explicitly name their argument do not affect
//! parameters which do not name an argument in terms of positional specifiers.
//!
//! A format string is required to use all of its arguments, otherwise it is a
//! compile-time error. You may refer to the same argument more than once in the
//! format string, although it must always be referred to with the same type.
//!
//! ## Named parameters
//!
//! Rust itself does not have a Python-like equivalent of named parameters to a
//! function, but the `format!` macro is a syntax extension which allows it to
//! leverage named parameters. Named parameters are listed at the end of the
//! argument list and have the syntax:
//!
//! ```text
//! identifier '=' expression
//! ```
//!
//! For example, the following `format!` expressions all use named argument:
//!
//! ```
//! format!("{argument}", argument = "test");   // => "test"
//! format!("{name} {}", 1, name = 2);        // => "2 1"
//! format!("{a} {c} {b}", a="a", b='b', c=3);  // => "a 3 b"
//! ```
//!
//! It is not valid to put positional parameters (those without names) after
//! arguments which have names. Like with positional parameters, it is not
//! valid to provide named parameters that are unused by the format string.
//!
//! ## Argument types
//!
//! Each argument's type is dictated by the format string. It is a requirement
//! that every argument is only ever referred to by one type. For example, this
//! is an invalid format string:
//!
//! ```text
//! {0:x} {0:o}
//! ```
//!
//! This is invalid because the first argument is both referred to as a
//! hexadecimal as well as an
//! octal.
//!
//! There are various parameters which do require a particular type, however.
//! Namely, the `{:.*}` syntax, which sets the number of numbers after the
//! decimal in floating-point types:
//!
//! ```
//! let formatted_number = format!("{:.*}", 2, 1.234567);
//!
//! assert_eq!("1.23", formatted_number)
//! ```
//!
//! If this syntax is used, then the number of characters to print precedes the
//! actual object being formatted, and the number of characters must have the
//! type `usize`. Although a `usize` can be printed with `{}`, it is invalid to
//! reference an argument as such. For example this is another invalid format
//! string:
//!
//! ```text
//! {:.*} {0}
//! ```
//!
//! ## Formatting traits
//!
//! When requesting that an argument be formatted with a particular type, you
//! are actually requesting that an argument ascribes to a particular trait.
//! This allows multiple actual types to be formatted via `{:x}` (like `i8` as
//! well as `isize`).  The current mapping of types to traits is:
//!
//! * *nothing* ⇒ [`Display`](trait.Display.html)
//! * `?` ⇒ [`Debug`](trait.Debug.html)
//! * `o` ⇒ [`Octal`](trait.Octal.html)
//! * `x` ⇒ [`LowerHex`](trait.LowerHex.html)
//! * `X` ⇒ [`UpperHex`](trait.UpperHex.html)
//! * `p` ⇒ [`Pointer`](trait.Pointer.html)
//! * `b` ⇒ [`Binary`](trait.Binary.html)
//! * `e` ⇒ [`LowerExp`](trait.LowerExp.html)
//! * `E` ⇒ [`UpperExp`](trait.UpperExp.html)
//!
//! What this means is that any type of argument which implements the
//! `fmt::Binary` trait can then be formatted with `{:b}`. Implementations
//! are provided for these traits for a number of primitive types by the
//! standard library as well. If no format is specified (as in `{}` or `{:6}`),
//! then the format trait used is the `Display` trait.
//!
//! When implementing a format trait for your own type, you will have to
//! implement a method of the signature:
//!
//! ```
//! # #![allow(dead_code)]
//! # use std::fmt;
//! # struct Foo; // our custom type
//! # impl fmt::Display for Foo {
//! fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//! # write!(f, "testing, testing")
//! # } }
//! ```
//!
//! Your type will be passed as `self` by-reference, and then the function
//! should emit output into the `f.buf` stream. It is up to each format trait
//! implementation to correctly adhere to the requested formatting parameters.
//! The values of these parameters will be listed in the fields of the
//! `Formatter` struct. In order to help with this, the `Formatter` struct also
//! provides some helper methods.
//!
//! Additionally, the return value of this function is `fmt::Result` which is a
//! typedef to `Result<(), std::io::Error>` (also known as `std::io::Result<()>`).
//! Formatting implementations should ensure that they return errors from `write!`
//! correctly (propagating errors upward).
//!
//! An example of implementing the formatting traits would look
//! like:
//!
//! ```
//! use std::fmt;
//!
//! #[derive(Debug)]
//! struct Vector2D {
//!     x: isize,
//!     y: isize,
//! }
//!
//! impl fmt::Display for Vector2D {
//!     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//!         // The `f` value implements the `Write` trait, which is what the
//!         // write! macro is expecting. Note that this formatting ignores the
//!         // various flags provided to format strings.
//!         write!(f, "({}, {})", self.x, self.y)
//!     }
//! }
//!
//! // Different traits allow different forms of output of a type. The meaning
//! // of this format is to print the magnitude of a vector.
//! impl fmt::Binary for Vector2D {
//!     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//!         let magnitude = (self.x * self.x + self.y * self.y) as f64;
//!         let magnitude = magnitude.sqrt();
//!
//!         // Respect the formatting flags by using the helper method
//!         // `pad_integral` on the Formatter object. See the method
//!         // documentation for details, and the function `pad` can be used
//!         // to pad strings.
//!         let decimals = f.precision().unwrap_or(3);
//!         let string = format!("{:.*}", decimals, magnitude);
//!         f.pad_integral(true, "", &string)
//!     }
//! }
//!
//! fn main() {
//!     let myvector = Vector2D { x: 3, y: 4 };
//!
//!     println!("{}", myvector);       // => "(3, 4)"
//!     println!("{:?}", myvector);     // => "Vector2D {x: 3, y:4}"
//!     println!("{:10.3b}", myvector); // => "     5.000"
//! }
//! ```
//!
//! ### `fmt::Display` vs `fmt::Debug`
//!
//! These two formatting traits have distinct purposes:
//!
//! - `fmt::Display` implementations assert that the type can be faithfully
//!   represented as a UTF-8 string at all times. It is **not** expected that
//!   all types implement the `Display` trait.
//! - `fmt::Debug` implementations should be implemented for **all** public types.
//!   Output will typically represent the internal state as faithfully as possible.
//!   The purpose of the `Debug` trait is to facilitate debugging Rust code. In
//!   most cases, using `#[derive(Debug)]` is sufficient and recommended.
//!
//! Some examples of the output from both traits:
//!
//! ```
//! assert_eq!(format!("{} {:?}", 3, 4), "3 4");
//! assert_eq!(format!("{} {:?}", 'a', 'b'), "a 'b'");
//! assert_eq!(format!("{} {:?}", "foo\n", "bar\n"), "foo\n \"bar\\n\"");
//! ```
//!
//! ## Related macros
//!
//! There are a number of related macros in the `format!` family. The ones that
//! are currently implemented are:
//!
//! ```ignore
//! format!      // described above
//! write!       // first argument is a &mut io::Write, the destination
//! writeln!     // same as write but appends a newline
//! print!       // the format string is printed to the standard output
//! println!     // same as print but appends a newline
//! format_args! // described below.
//! ```
//!
//! ### `write!`
//!
//! This and `writeln` are two macros which are used to emit the format string
//! to a specified stream. This is used to prevent intermediate allocations of
//! format strings and instead directly write the output. Under the hood, this
//! function is actually invoking the `write` function defined in this module.
//! Example usage is:
//!
//! ```
//! # #![allow(unused_must_use)]
//! use std::io::Write;
//! let mut w = Vec::new();
//! write!(&mut w, "Hello {}!", "world");
//! ```
//!
//! ### `print!`
//!
//! This and `println` emit their output to stdout. Similarly to the `write!`
//! macro, the goal of these macros is to avoid intermediate allocations when
//! printing output. Example usage is:
//!
//! ```
//! print!("Hello {}!", "world");
//! println!("I have a newline {}", "character at the end");
//! ```
//!
//! ### `format_args!`
//!
//! This is a curious macro which is used to safely pass around
//! an opaque object describing the format string. This object
//! does not require any heap allocations to create, and it only
//! references information on the stack. Under the hood, all of
//! the related macros are implemented in terms of this. First
//! off, some example usage is:
//!
//! ```
//! # #![allow(unused_must_use)]
//! use std::fmt;
//! use std::io::{self, Write};
//!
//! fmt::format(format_args!("this returns {}", "String"));
//!
//! let mut some_writer = io::stdout();
//! write!(&mut some_writer, "{}", format_args!("print with a {}", "macro"));
//!
//! fn my_fmt_fn(args: fmt::Arguments) {
//!     write!(&mut io::stdout(), "{}", args);
//! }
//! my_fmt_fn(format_args!("or a {} too", "function"));
//! ```
//!
//! The result of the `format_args!` macro is a value of type `fmt::Arguments`.
//! This structure can then be passed to the `write` and `format` functions
//! inside this module in order to process the format string.
//! The goal of this macro is to even further prevent intermediate allocations
//! when dealing formatting strings.
//!
//! For example, a logging library could use the standard formatting syntax, but
//! it would internally pass around this structure until it has been determined
//! where output should go to.
//!
//! # Syntax
//!
//! The syntax for the formatting language used is drawn from other languages,
//! so it should not be too alien. Arguments are formatted with python-like
//! syntax, meaning that arguments are surrounded by `{}` instead of the C-like
//! `%`. The actual grammar for the formatting syntax is:
//!
//! ```text
//! format_string := <text> [ format <text> ] *
//! format := '{' [ argument ] [ ':' format_spec ] '}'
//! argument := integer | identifier
//!
//! format_spec := [[fill]align][sign]['#'][0][width]['.' precision][type]
//! fill := character
//! align := '<' | '^' | '>'
//! sign := '+' | '-'
//! width := count
//! precision := count | '*'
//! type := identifier | ''
//! count := parameter | integer
//! parameter := integer '$'
//! ```
//!
//! # Formatting Parameters
//!
//! Each argument being formatted can be transformed by a number of formatting
//! parameters (corresponding to `format_spec` in the syntax above). These
//! parameters affect the string representation of what's being formatted. This
//! syntax draws heavily from Python's, so it may seem a bit familiar.
//!
//! ## Fill/Alignment
//!
//! The fill character is provided normally in conjunction with the `width`
//! parameter. This indicates that if the value being formatted is smaller than
//! `width` some extra characters will be printed around it. The extra
//! characters are specified by `fill`, and the alignment can be one of two
//! options:
//!
//! * `<` - the argument is left-aligned in `width` columns
//! * `^` - the argument is center-aligned in `width` columns
//! * `>` - the argument is right-aligned in `width` columns
//!
//! Note that alignment may not be implemented by some types. A good way
//! to ensure padding is applied is to format your input, then use this
//! resulting string to pad your output.
//!
//! ## Sign/`#`/`0`
//!
//! These can all be interpreted as flags for a particular formatter.
//!
//! * `+` - This is intended for numeric types and indicates that the sign
//!         should always be printed. Positive signs are never printed by
//!         default, and the negative sign is only printed by default for the
//!         `Signed` trait. This flag indicates that the correct sign (`+` or `-`)
//!         should always be printed.
//! * `-` - Currently not used
//! * `#` - This flag is indicates that the "alternate" form of printing should
//!         be used. The alternate forms are:
//!     * `#?` - pretty-print the `Debug` formatting
//!     * `#x` - precedes the argument with a `0x`
//!     * `#X` - precedes the argument with a `0x`
//!     * `#b` - precedes the argument with a `0b`
//!     * `#o` - precedes the argument with a `0o`
//! * `0` - This is used to indicate for integer formats that the padding should
//!         both be done with a `0` character as well as be sign-aware. A format
//!         like `{:08}` would yield `00000001` for the integer `1`, while the
//!         same format would yield `-0000001` for the integer `-1`. Notice that
//!         the negative version has one fewer zero than the positive version.
//!
//! ## Width
//!
//! This is a parameter for the "minimum width" that the format should take up.
//! If the value's string does not fill up this many characters, then the
//! padding specified by fill/alignment will be used to take up the required
//! space.
//!
//! The default fill/alignment for non-numerics is a space and left-aligned. The
//! defaults for numeric formatters is also a space but with right-alignment. If
//! the `0` flag is specified for numerics, then the implicit fill character is
//! `0`.
//!
//! The value for the width can also be provided as a `usize` in the list of
//! parameters by using the `2$` syntax indicating that the second argument is a
//! `usize` specifying the width.
//!
//! ## Precision
//!
//! For non-numeric types, this can be considered a "maximum width". If the resulting string is
//! longer than this width, then it is truncated down to this many characters and only those are
//! emitted.
//!
//! For integral types, this is ignored.
//!
//! For floating-point types, this indicates how many digits after the decimal point should be
//! printed.
//!
//! There are three possible ways to specify the desired `precision`:
//!
//! 1. An integer `.N`:
//!
//!    the integer `N` itself is the precision.
//!
//! 2. An integer followed by dollar sign `.N$`:
//!
//!    use format *argument* `N` (which must be a `usize`) as the precision.
//!
//! 3. An asterisk `.*`:
//!
//!    `.*` means that this `{...}` is associated with *two* format inputs rather than one: the
//!    first input holds the `usize` precision, and the second holds the value to print.  Note that
//!    in this case, if one uses the format string `{<arg>:<spec>.*}`, then the `<arg>` part refers
//!    to the *value* to print, and the `precision` must come in the input preceding `<arg>`.
//!
//! For example, these:
//!
//! ```
//! // Hello {arg 0 (x)} is {arg 1 (0.01} with precision specified inline (5)}
//! println!("Hello {0} is {1:.5}", "x", 0.01);
//!
//! // Hello {arg 1 (x)} is {arg 2 (0.01} with precision specified in arg 0 (5)}
//! println!("Hello {1} is {2:.0$}", 5, "x", 0.01);
//!
//! // Hello {arg 0 (x)} is {arg 2 (0.01} with precision specified in arg 1 (5)}
//! println!("Hello {0} is {2:.1$}", "x", 5, 0.01);
//!
//! // Hello {next arg (x)} is {second of next two args (0.01} with precision
//! //                          specified in first of next two args (5)}
//! println!("Hello {} is {:.*}",    "x", 5, 0.01);
//!
//! // Hello {next arg (x)} is {arg 2 (0.01} with precision
//! //                          specified in its predecessor (5)}
//! println!("Hello {} is {2:.*}",   "x", 5, 0.01);
//! ```
//!
//! All print the same thing:
//!
//! ```text
//! Hello x is 0.01000
//! ```
//!
//! While these:
//!
//! ```
//! println!("{}, `{name:.*}` has 3 fractional digits", "Hello", 3, name=1234.56);
//! println!("{}, `{name:.*}` has 3 characters", "Hello", 3, name="1234.56");
//! ```
//!
//! print two significantly different things:
//!
//! ```text
//! Hello, `1234.560` has 3 fractional digits
//! Hello, `123` has 3 characters
//! ```
//!
//! # Escaping
//!
//! The literal characters `{` and `}` may be included in a string by preceding
//! them with the same character. For example, the `{` character is escaped with
//! `{{` and the `}` character is escaped with `}}`.

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Formatter, Result, Write, rt};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Octal, Binary};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Display, Debug};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{LowerHex, UpperHex, Pointer};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{LowerExp, UpperExp};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::Error;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{ArgumentV1, Arguments, write, radix, Radix, RadixFmt};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{DebugList, DebugMap, DebugSet, DebugStruct, DebugTuple};

use string;

/// The format function takes a precompiled format string and a list of
/// arguments, to return the resulting formatted string.
///
/// # Arguments
///
///   * args - a structure of arguments generated via the `format_args!` macro.
///
/// # Examples
///
/// ```
/// use std::fmt;
///
/// let s = fmt::format(format_args!("Hello, {}!", "world"));
/// assert_eq!(s, "Hello, world!".to_string());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn format(args: Arguments) -> string::String {
    let mut output = string::String::new();
    let _ = output.write_fmt(args);
    output
}
