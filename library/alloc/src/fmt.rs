//! Utilities for formatting and printing `String`s.
//!
//! This module contains the runtime support for the [`format!`] syntax extension.
//! This macro is implemented in the compiler to emit calls to this module in
//! order to format arguments at runtime into strings.
//!
//! # Usage
//!
//! The [`format!`] macro is intended to be familiar to those coming from C's
//! `printf`/`fprintf` functions or Python's `str.format` function.
//!
//! Some examples of the [`format!`] extension are:
//!
//! ```
//! # #![allow(unused_must_use)]
//! format!("Hello");                 // => "Hello"
//! format!("Hello, {}!", "world");   // => "Hello, world!"
//! format!("The number is {}", 1);   // => "The number is 1"
//! format!("{:?}", (3, 4));          // => "(3, 4)"
//! format!("{value}", value=4);      // => "4"
//! let people = "Rustaceans";
//! format!("Hello {people}!");       // => "Hello Rustaceans!"
//! format!("{} {}", 1, 2);           // => "1 2"
//! format!("{:04}", 42);             // => "0042" with leading zeros
//! format!("{:#?}", (100, 200));     // => "(
//!                                   //       100,
//!                                   //       200,
//!                                   //     )"
//! ```
//!
//! From these, you can see that the first argument is a format string. It is
//! required by the compiler for this to be a string literal; it cannot be a
//! variable passed in (in order to perform validity checking). The compiler
//! will then parse the format string and determine if the list of arguments
//! provided is suitable to pass to this format string.
//!
//! To convert a single value to a string, use the [`to_string`] method. This
//! will use the [`Display`] formatting trait.
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
//! # #![allow(unused_must_use)]
//! format!("{1} {} {0} {}", 1, 2); // => "2 1 1 2"
//! ```
//!
//! The internal iterator over the argument has not been advanced by the time
//! the first `{}` is seen, so it prints the first argument. Then upon reaching
//! the second `{}`, the iterator has advanced forward to the second argument.
//! Essentially, parameters that explicitly name their argument do not affect
//! parameters that do not name an argument in terms of positional specifiers.
//!
//! A format string is required to use all of its arguments, otherwise it is a
//! compile-time error. You may refer to the same argument more than once in the
//! format string.
//!
//! ## Named parameters
//!
//! Rust itself does not have a Python-like equivalent of named parameters to a
//! function, but the [`format!`] macro is a syntax extension that allows it to
//! leverage named parameters. Named parameters are listed at the end of the
//! argument list and have the syntax:
//!
//! ```text
//! identifier '=' expression
//! ```
//!
//! For example, the following [`format!`] expressions all use named arguments:
//!
//! ```
//! # #![allow(unused_must_use)]
//! format!("{argument}", argument = "test");   // => "test"
//! format!("{name} {}", 1, name = 2);          // => "2 1"
//! format!("{a} {c} {b}", a="a", b='b', c=3);  // => "a 3 b"
//! ```
//!
//! If a named parameter does not appear in the argument list, `format!` will
//! reference a variable with that name in the current scope.
//!
//! ```
//! # #![allow(unused_must_use)]
//! let argument = 2 + 2;
//! format!("{argument}");   // => "4"
//!
//! fn make_string(a: u32, b: &str) -> String {
//!     format!("{b} {a}")
//! }
//! make_string(927, "label"); // => "label 927"
//! ```
//!
//! It is not valid to put positional parameters (those without names) after
//! arguments that have names. Like with positional parameters, it is not
//! valid to provide named parameters that are unused by the format string.
//!
//! # Formatting Parameters
//!
//! Each argument being formatted can be transformed by a number of formatting
//! parameters (corresponding to `format_spec` in [the syntax](#syntax)). These
//! parameters affect the string representation of what's being formatted.
//!
//! The colon `:` in format syntax divides indentifier of the input data and
//! the formatting options, the colon itself does not change anything, only
//! introduces the options.
//!
//! ```
//! let a = 5;
//! let b = &a;
//! println!("{a:e} {b:p}"); // => 5e0 0x7ffe37b7273c
//! ```
//!
//! ## Width
//!
//! ```
//! // All of these print "Hello x    !"
//! println!("Hello {:5}!", "x");
//! println!("Hello {:1$}!", "x", 5);
//! println!("Hello {1:0$}!", 5, "x");
//! println!("Hello {:width$}!", "x", width = 5);
//! let width = 5;
//! println!("Hello {:width$}!", "x");
//! ```
//!
//! This is a parameter for the "minimum width" that the format should take up.
//! If the value's string does not fill up this many characters, then the
//! padding specified by fill/alignment will be used to take up the required
//! space (see below).
//!
//! The value for the width can also be provided as a [`usize`] in the list of
//! parameters by adding a postfix `$`, indicating that the second argument is
//! a [`usize`] specifying the width.
//!
//! Referring to an argument with the dollar syntax does not affect the "next
//! argument" counter, so it's usually a good idea to refer to arguments by
//! position, or use named arguments.
//!
//! ## Fill/Alignment
//!
//! ```
//! assert_eq!(format!("Hello {:<5}!", "x"),  "Hello x    !");
//! assert_eq!(format!("Hello {:-<5}!", "x"), "Hello x----!");
//! assert_eq!(format!("Hello {:^5}!", "x"),  "Hello   x  !");
//! assert_eq!(format!("Hello {:>5}!", "x"),  "Hello     x!");
//! ```
//!
//! The optional fill character and alignment is provided normally in conjunction with the
//! [`width`](#width) parameter. It must be defined before `width`, right after the `:`.
//! This indicates that if the value being formatted is smaller than
//! `width` some extra characters will be printed around it.
//! Filling comes in the following variants for different alignments:
//!
//! * `[fill]<` - the argument is left-aligned in `width` columns
//! * `[fill]^` - the argument is center-aligned in `width` columns
//! * `[fill]>` - the argument is right-aligned in `width` columns
//!
//! The default [fill/alignment](#fillalignment) for non-numerics is a space and
//! left-aligned. The
//! default for numeric formatters is also a space character but with right-alignment. If
//! the `0` flag (see below) is specified for numerics, then the implicit fill character is
//! `0`.
//!
//! Note that alignment might not be implemented by some types. In particular, it
//! is not generally implemented for the `Debug` trait.  A good way to ensure
//! padding is applied is to format your input, then pad this resulting string
//! to obtain your output:
//!
//! ```
//! println!("Hello {:^15}!", format!("{:?}", Some("hi"))); // => "Hello   Some("hi")   !"
//! ```
//!
//! ## Sign/`#`/`0`
//!
//! ```
//! assert_eq!(format!("Hello {:+}!", 5), "Hello +5!");
//! assert_eq!(format!("{:#x}!", 27), "0x1b!");
//! assert_eq!(format!("Hello {:05}!", 5),  "Hello 00005!");
//! assert_eq!(format!("Hello {:05}!", -5), "Hello -0005!");
//! assert_eq!(format!("{:#010x}!", 27), "0x0000001b!");
//! ```
//!
//! These are all flags altering the behavior of the formatter.
//!
//! * `+` - This is intended for numeric types and indicates that the sign
//!         should always be printed. By default only the negative sign of signed values
//!         is printed, and the sign of positive or unsigned values is omitted.
//!         This flag indicates that the correct sign (`+` or `-`) should always be printed.
//! * `-` - Currently not used
//! * `#` - This flag indicates that the "alternate" form of printing should
//!         be used. The alternate forms are:
//!     * `#?` - pretty-print the [`Debug`] formatting (adds linebreaks and indentation)
//!     * `#x` - precedes the argument with a `0x`
//!     * `#X` - precedes the argument with a `0x`
//!     * `#b` - precedes the argument with a `0b`
//!     * `#o` - precedes the argument with a `0o`
//!
//!   See [Formatting traits](#formatting-traits) for a description of what the `?`, `x`, `X`,
//!   `b`, and `o` flags do.
//!
//! * `0` - This is used to indicate for integer formats that the padding to `width` should
//!         both be done with a `0` character as well as be sign-aware. A format
//!         like `{:08}` would yield `00000001` for the integer `1`, while the
//!         same format would yield `-0000001` for the integer `-1`. Notice that
//!         the negative version has one fewer zero than the positive version.
//!         Note that padding zeros are always placed after the sign (if any)
//!         and before the digits. When used together with the `#` flag, a similar
//!         rule applies: padding zeros are inserted after the prefix but before
//!         the digits. The prefix is included in the total width.
//!         This flag overrides the [fill character and alignment flag](#fillalignment).
//!
//! ## Precision
//!
//! For non-numeric types, this can be considered a "maximum width". If the resulting string is
//! longer than this width, then it is truncated down to this many characters and that truncated
//! value is emitted with proper `fill`, `alignment` and `width` if those parameters are set.
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
//! 2. An integer or name followed by dollar sign `.N$`:
//!
//!    use format *argument* `N` (which must be a `usize`) as the precision.
//!
//! 3. An asterisk `.*`:
//!
//!    `.*` means that this `{...}` is associated with *two* format inputs rather than one:
//!    - If a format string in the fashion of `{:<spec>.*}` is used, then the first input holds
//!      the `usize` precision, and the second holds the value to print.
//!    - If a format string in the fashion of `{<arg>:<spec>.*}` is used, then the `<arg>` part
//!      refers to the value to print, and the `precision` is taken like it was specified with an
//!      omitted positional parameter (`{}` instead of `{<arg>:}`).
//!
//! For example, the following calls all print the same thing `Hello x is 0.01000`:
//!
//! ```
//! // Hello {arg 0 ("x")} is {arg 1 (0.01) with precision specified inline (5)}
//! println!("Hello {0} is {1:.5}", "x", 0.01);
//!
//! // Hello {arg 1 ("x")} is {arg 2 (0.01) with precision specified in arg 0 (5)}
//! println!("Hello {1} is {2:.0$}", 5, "x", 0.01);
//!
//! // Hello {arg 0 ("x")} is {arg 2 (0.01) with precision specified in arg 1 (5)}
//! println!("Hello {0} is {2:.1$}", "x", 5, 0.01);
//!
//! // Hello {next arg -> arg 0 ("x")} is {second of next two args -> arg 2 (0.01) with precision
//! //                          specified in first of next two args -> arg 1 (5)}
//! println!("Hello {} is {:.*}",    "x", 5, 0.01);
//!
//! // Hello {arg 1 ("x")} is {arg 2 (0.01) with precision
//! //                          specified in next arg -> arg 0 (5)}
//! println!("Hello {1} is {2:.*}",  5, "x", 0.01);
//!
//! // Hello {next arg -> arg 0 ("x")} is {arg 2 (0.01) with precision
//! //                          specified in next arg -> arg 1 (5)}
//! println!("Hello {} is {2:.*}",   "x", 5, 0.01);
//!
//! // Hello {next arg -> arg 0 ("x")} is {arg "number" (0.01) with precision specified
//! //                          in arg "prec" (5)}
//! println!("Hello {} is {number:.prec$}", "x", prec = 5, number = 0.01);
//! ```
//!
//! While these:
//!
//! ```
//! println!("{}, `{name:.*}` has 3 fractional digits", "Hello", 3, name=1234.56);
//! println!("{}, `{name:.*}` has 3 characters", "Hello", 3, name="1234.56");
//! println!("{}, `{name:>8.*}` has 3 right-aligned characters", "Hello", 3, name="1234.56");
//! ```
//!
//! print three significantly different things:
//!
//! ```text
//! Hello, `1234.560` has 3 fractional digits
//! Hello, `123` has 3 characters
//! Hello, `     123` has 3 right-aligned characters
//! ```
//!
//! When truncating these values, Rust uses [round half-to-even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even),
//! which is the default rounding mode in IEEE 754.
//! For example,
//!
//! ```
//! print!("{0:.1$e}", 12345, 3);
//! print!("{0:.1$e}", 12355, 3);
//! ```
//!
//! Would return:
//!
//! ```text
//! 1.234e4
//! 1.236e4
//! ```
//!
//! ## Localization
//!
//! In some programming languages, the behavior of string formatting functions
//! depends on the operating system's locale setting. The format functions
//! provided by Rust's standard library do not have any concept of locale and
//! will produce the same results on all systems regardless of user
//! configuration.
//!
//! For example, the following code will always print `1.5` even if the system
//! locale uses a decimal separator other than a dot.
//!
//! ```
//! println!("The value is {}", 1.5);
//! ```
//!
//! # Escaping
//!
//! The literal characters `{` and `}` may be included in a string by preceding
//! them with the same character. For example, the `{` character is escaped with
//! `{{` and the `}` character is escaped with `}}`.
//!
//! ```
//! assert_eq!(format!("Hello {{}}"), "Hello {}");
//! assert_eq!(format!("{{ Hello"), "{ Hello");
//! ```
//!
//! # Syntax
//!
//! To summarize, here you can find the full grammar of format strings.
//! The syntax for the formatting language used is drawn from other languages,
//! so it should not be too alien. Arguments are formatted with Python-like
//! syntax, meaning that arguments are surrounded by `{}` instead of the C-like
//! `%`. The actual grammar for the formatting syntax is:
//!
//! ```text
//! format_string := text [ maybe_format text ] *
//! maybe_format := '{' '{' | '}' '}' | format
//! format := '{' [ argument ] [ ':' format_spec ] [ ws ] * '}'
//! argument := integer | identifier
//!
//! format_spec := [[fill]align][sign]['#']['0'][width]['.' precision]type
//! fill := character
//! align := '<' | '^' | '>'
//! sign := '+' | '-'
//! width := count
//! precision := count | '*'
//! type := '' | '?' | 'x?' | 'X?' | identifier
//! count := parameter | integer
//! parameter := argument '$'
//! ```
//! In the above grammar,
//! - `text` must not contain any `'{'` or `'}'` characters,
//! - `ws` is any character for which [`char::is_whitespace`] returns `true`, has no semantic
//!   meaning and is completely optional,
//! - `integer` is a decimal integer that may contain leading zeroes and must fit into an `usize` and
//! - `identifier` is an `IDENTIFIER_OR_KEYWORD` (not an `IDENTIFIER`) as defined by the [Rust language reference](https://doc.rust-lang.org/reference/identifiers.html).
//!
//! # Formatting traits
//!
//! When requesting that an argument be formatted with a particular type, you
//! are actually requesting that an argument ascribes to a particular trait.
//! This allows multiple actual types to be formatted via `{:x}` (like [`i8`] as
//! well as [`isize`]). The current mapping of types to traits is:
//!
//! * *nothing* ⇒ [`Display`]
//! * `?` ⇒ [`Debug`]
//! * `x?` ⇒ [`Debug`] with lower-case hexadecimal integers
//! * `X?` ⇒ [`Debug`] with upper-case hexadecimal integers
//! * `o` ⇒ [`Octal`]
//! * `x` ⇒ [`LowerHex`]
//! * `X` ⇒ [`UpperHex`]
//! * `p` ⇒ [`Pointer`]
//! * `b` ⇒ [`Binary`]
//! * `e` ⇒ [`LowerExp`]
//! * `E` ⇒ [`UpperExp`]
//!
//! What this means is that any type of argument which implements the
//! [`fmt::Binary`][`Binary`] trait can then be formatted with `{:b}`. Implementations
//! are provided for these traits for a number of primitive types by the
//! standard library as well. If no format is specified (as in `{}` or `{:6}`),
//! then the format trait used is the [`Display`] trait.
//!
//! When implementing a format trait for your own type, you will have to
//! implement a method of the signature:
//!
//! ```
//! # #![allow(dead_code)]
//! # use std::fmt;
//! # struct Foo; // our custom type
//! # impl fmt::Display for Foo {
//! fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//! # write!(f, "testing, testing")
//! # } }
//! ```
//!
//! Your type will be passed as `self` by-reference, and then the function
//! should emit output into the Formatter `f` which implements `fmt::Write`. It is up to each
//! format trait implementation to correctly adhere to the requested formatting parameters.
//! The values of these parameters can be accessed with methods of the
//! [`Formatter`] struct. In order to help with this, the [`Formatter`] struct also
//! provides some helper methods.
//!
//! Additionally, the return value of this function is [`fmt::Result`] which is a
//! type alias of <code>[Result]<(), [std::fmt::Error]></code>. Formatting implementations
//! should ensure that they propagate errors from the [`Formatter`] (e.g., when
//! calling [`write!`]). However, they should never return errors spuriously. That
//! is, a formatting implementation must and may only return an error if the
//! passed-in [`Formatter`] returns an error. This is because, contrary to what
//! the function signature might suggest, string formatting is an infallible
//! operation. This function only returns a [`Result`] because writing to the
//! underlying stream might fail and it must provide a way to propagate the fact
//! that an error has occurred back up the stack.
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
//!     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
//!     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//!         let magnitude = (self.x * self.x + self.y * self.y) as f64;
//!         let magnitude = magnitude.sqrt();
//!
//!         // Respect the formatting flags by using the helper method
//!         // `pad_integral` on the Formatter object. See the method
//!         // documentation for details, and the function `pad` can be used
//!         // to pad strings.
//!         let decimals = f.precision().unwrap_or(3);
//!         let string = format!("{magnitude:.decimals$}");
//!         f.pad_integral(true, "", &string)
//!     }
//! }
//!
//! fn main() {
//!     let myvector = Vector2D { x: 3, y: 4 };
//!
//!     println!("{myvector}");       // => "(3, 4)"
//!     println!("{myvector:?}");     // => "Vector2D {x: 3, y:4}"
//!     println!("{myvector:10.3b}"); // => "     5.000"
//! }
//! ```
//!
//! ### `fmt::Display` vs `fmt::Debug`
//!
//! These two formatting traits have distinct purposes:
//!
//! - [`fmt::Display`][`Display`] implementations assert that the type can be faithfully
//!   represented as a UTF-8 string at all times. It is **not** expected that
//!   all types implement the [`Display`] trait.
//! - [`fmt::Debug`][`Debug`] implementations should be implemented for **all** public types.
//!   Output will typically represent the internal state as faithfully as possible.
//!   The purpose of the [`Debug`] trait is to facilitate debugging Rust code. In
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
//! # Related macros
//!
//! There are a number of related macros in the [`format!`] family. The ones that
//! are currently implemented are:
//!
//! ```ignore (only-for-syntax-highlight)
//! format!      // described above
//! write!       // first argument is either a &mut io::Write or a &mut fmt::Write, the destination
//! writeln!     // same as write but appends a newline
//! print!       // the format string is printed to the standard output
//! println!     // same as print but appends a newline
//! eprint!      // the format string is printed to the standard error
//! eprintln!    // same as eprint but appends a newline
//! format_args! // described below.
//! ```
//!
//! ### `write!`
//!
//! [`write!`] and [`writeln!`] are two macros which are used to emit the format string
//! to a specified stream. This is used to prevent intermediate allocations of
//! format strings and instead directly write the output. Under the hood, this
//! function is actually invoking the [`write_fmt`] function defined on the
//! [`std::io::Write`] and the [`std::fmt::Write`] trait. Example usage is:
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
//! This and [`println!`] emit their output to stdout. Similarly to the [`write!`]
//! macro, the goal of these macros is to avoid intermediate allocations when
//! printing output. Example usage is:
//!
//! ```
//! print!("Hello {}!", "world");
//! println!("I have a newline {}", "character at the end");
//! ```
//! ### `eprint!`
//!
//! The [`eprint!`] and [`eprintln!`] macros are identical to
//! [`print!`] and [`println!`], respectively, except they emit their
//! output to stderr.
//!
//! ### `format_args!`
//!
//! [`format_args!`] is a curious macro used to safely pass around
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
//! let mut some_writer = io::stdout();
//! write!(&mut some_writer, "{}", format_args!("print with a {}", "macro"));
//!
//! fn my_fmt_fn(args: fmt::Arguments<'_>) {
//!     write!(&mut io::stdout(), "{args}");
//! }
//! my_fmt_fn(format_args!(", or a {} too", "function"));
//! ```
//!
//! The result of the [`format_args!`] macro is a value of type [`fmt::Arguments`].
//! This structure can then be passed to the [`write`] and [`format`] functions
//! inside this module in order to process the format string.
//! The goal of this macro is to even further prevent intermediate allocations
//! when dealing with formatting strings.
//!
//! For example, a logging library could use the standard formatting syntax, but
//! it would internally pass around this structure until it has been determined
//! where output should go to.
//!
//! [`fmt::Result`]: Result "fmt::Result"
//! [Result]: core::result::Result "std::result::Result"
//! [std::fmt::Error]: Error "fmt::Error"
//! [`write`]: write() "fmt::write"
//! [`to_string`]: crate::string::ToString::to_string "ToString::to_string"
//! [`write_fmt`]: ../../std/io/trait.Write.html#method.write_fmt
//! [`std::io::Write`]: ../../std/io/trait.Write.html
//! [`std::fmt::Write`]: ../../std/fmt/trait.Write.html
//! [`print!`]: ../../std/macro.print.html "print!"
//! [`println!`]: ../../std/macro.println.html "println!"
//! [`eprint!`]: ../../std/macro.eprint.html "eprint!"
//! [`eprintln!`]: ../../std/macro.eprintln.html "eprintln!"
//! [`format_args!`]: ../../std/macro.format_args.html "format_args!"
//! [`fmt::Arguments`]: Arguments "fmt::Arguments"
//! [`format`]: format() "fmt::format"

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "fmt_flags_align", since = "1.28.0")]
pub use core::fmt::Alignment;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::Error;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Arguments, write};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Binary, Octal};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Debug, Display};
#[unstable(feature = "formatting_options", issue = "118117")]
pub use core::fmt::{DebugAsHex, FormattingOptions, Sign};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{DebugList, DebugMap, DebugSet, DebugStruct, DebugTuple};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{Formatter, Result, Write};
#[unstable(feature = "debug_closure_helpers", issue = "117729")]
pub use core::fmt::{FromFn, from_fn};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{LowerExp, UpperExp};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::fmt::{LowerHex, Pointer, UpperHex};

#[cfg(not(no_global_oom_handling))]
use crate::string;

/// Takes an [`Arguments`] struct and returns the resulting formatted string.
///
/// The [`Arguments`] instance can be created with the [`format_args!`] macro.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::fmt;
///
/// let s = fmt::format(format_args!("Hello, {}!", "world"));
/// assert_eq!(s, "Hello, world!");
/// ```
///
/// Please note that using [`format!`] might be preferable.
/// Example:
///
/// ```
/// let s = format!("Hello, {}!", "world");
/// assert_eq!(s, "Hello, world!");
/// ```
///
/// [`format_args!`]: core::format_args
/// [`format!`]: crate::format
#[cfg(not(no_global_oom_handling))]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[inline]
pub fn format(args: Arguments<'_>) -> string::String {
    fn format_inner(args: Arguments<'_>) -> string::String {
        let capacity = args.estimated_capacity();
        let mut output = string::String::with_capacity(capacity);
        output
            .write_fmt(args)
            .expect("a formatting trait implementation returned an error when the underlying stream did not");
        output
    }

    args.as_str().map_or_else(|| format_inner(args), crate::borrow::ToOwned::to_owned)
}
