// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Utilities for formatting and printing strings

This module contains the runtime support for the `format!` syntax extension.
This macro is implemented in the compiler to emit calls to this module in order
to format arguments at runtime into strings and streams.

The functions contained in this module should not normally be used in everyday
use cases of `format!`. The assumptions made by these functions are unsafe for
all inputs, and the compiler performs a large amount of validation on the
arguments to `format!` in order to ensure safety at runtime. While it is
possible to call these functions directly, it is not recommended to do so in the
general case.

## Usage

The `format!` macro is intended to be familiar to those coming from C's
printf/fprintf functions or Python's `str.format` function. In its current
revision, the `format!` macro returns a `~str` type which is the result of the
formatting. In the future it will also be able to pass in a stream to format
arguments directly while performing minimal allocations.

Some examples of the `format!` extension are:

```rust
format!("Hello");                 // => ~"Hello"
format!("Hello, {:s}!", "world"); // => ~"Hello, world!"
format!("The number is {:d}", 1); // => ~"The number is 1"
format!("{:?}", ~[3, 4]);         // => ~"~[3, 4]"
format!("{value}", value=4);      // => ~"4"
format!("{} {}", 1, 2);           // => ~"1 2"
```

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

Things can get a little tricky once you start intermingling the two types of
positional specifiers. The "next argument" specifier can be thought of as an
iterator over the argument. Each time a "next argument" specifier is seen, the
iterator advances. This leads to behavior like this:

```rust
format!("{1} {} {0} {}", 1, 2); // => ~"2 1 1 2"
```

The internal iterator over the argument has not been advanced by the time the
first `{}` is seen, so it prints the first argument. Then upon reaching the
second `{}`, the iterator has advanced forward to the second argument.
Essentially, parameters which explicitly name their argument do not affect
parameters which do not name an argument in terms of positional specifiers.

A format string is required to use all of its arguments, otherwise it is a
compile-time error. You may refer to the same argument more than once in the
format string, although it must always be referred to with the same type.

### Named parameters

Rust itself does not have a Python-like equivalent of named parameters to a
function, but the `format!` macro is a syntax extension which allows it to
leverage named parameters. Named parameters are listed at the end of the
argument list and have the syntax:

```ignore
identifier '=' expression
```

For example, the following `format!` expressions all use named argument:

```rust
format!("{argument}", argument = "test");       // => ~"test"
format!("{name} {}", 1, name = 2);              // => ~"2 1"
format!("{a:s} {c:d} {b:?}", a="a", b=(), c=3); // => ~"a 3 ()"
```

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

```ignore
{0:d} {0:s}
```

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

```ignore
{:.*s} {0:u}
```

### Formatting traits

When requesting that an argument be formatted with a particular type, you are
actually requesting that an argument ascribes to a particular trait. This allows
multiple actual types to be formatted via `{:d}` (like `i8` as well as `int`).
The current mapping of types to traits is:

* `?` ⇒ `Poly`
* `d` ⇒ `Signed`
* `i` ⇒ `Signed`
* `u` ⇒ `Unsigned`
* `b` ⇒ `Bool`
* `c` ⇒ `Char`
* `o` ⇒ `Octal`
* `x` ⇒ `LowerHex`
* `X` ⇒ `UpperHex`
* `s` ⇒ `String`
* `p` ⇒ `Pointer`
* `t` ⇒ `Binary`
* `f` ⇒ `Float`
* `e` ⇒ `LowerExp`
* `E` ⇒ `UpperExp`
* *nothing* ⇒ `Show`

What this means is that any type of argument which implements the
`std::fmt::Binary` trait can then be formatted with `{:t}`. Implementations are
provided for these traits for a number of primitive types by the standard
library as well. If no format is specified (as in `{}` or `{:6}`), then the
format trait used is the `Show` trait. This is one of the more commonly
implemented traits when formatting a custom type.

When implementing a format trait for your own type, you will have to implement a
method of the signature:

```rust
# use std;
# mod fmt { pub type Result = (); }
# struct T;
# trait SomeName<T> {
fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result;
# }
```

Your type will be passed as `self` by-reference, and then the function should
emit output into the `f.buf` stream. It is up to each format trait
implementation to correctly adhere to the requested formatting parameters. The
values of these parameters will be listed in the fields of the `Formatter`
struct. In order to help with this, the `Formatter` struct also provides some
helper methods.

Additionally, the return value of this function is `fmt::Result` which is a
typedef to `Result<(), IoError>` (also known as `IoError<()>`). Formatting
implementations should ensure that they return errors from `write!` correctly
(propagating errors upward).

An example of implementing the formatting traits would look
like:

```rust
use std::fmt;
use std::f64;

struct Vector2D {
    x: int,
    y: int,
}

impl fmt::Show for Vector2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f.buf` value is of the type `&mut io::Writer`, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        write!(f.buf, "({}, {})", self.x, self.y)
    }
}

// Different traits allow different forms of output of a type. The meaning of
// this format is to print the magnitude of a vector.
impl fmt::Binary for Vector2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let magnitude = (self.x * self.x + self.y * self.y) as f64;
        let magnitude = magnitude.sqrt();

        // Respect the formatting flags by using the helper method
        // `pad_integral` on the Formatter object. See the method documentation
        // for details, and the function `pad` can be used to pad strings.
        let decimals = f.precision.unwrap_or(3);
        let string = f64::to_str_exact(magnitude, decimals);
        f.pad_integral(string.as_bytes(), "", true)
    }
}

fn main() {
    let myvector = Vector2D { x: 3, y: 4 };

    println!("{}", myvector);       // => "(3, 4)"
    println!("{:10.3t}", myvector); // => "     5.000"
}
```

### Related macros

There are a number of related macros in the `format!` family. The ones that are
currently implemented are:

```rust,ignore
format!      // described above
write!       // first argument is a &mut io::Writer, the destination
writeln!     // same as write but appends a newline
print!       // the format string is printed to the standard output
println!     // same as print but appends a newline
format_args! // described below.
```


#### `write!`

This and `writeln` are two macros which are used to emit the format string to a
specified stream. This is used to prevent intermediate allocations of format
strings and instead directly write the output. Under the hood, this function is
actually invoking the `write` function defined in this module. Example usage is:

```rust
# #[allow(unused_must_use)];
use std::io;

let mut w = io::MemWriter::new();
write!(&mut w as &mut io::Writer, "Hello {}!", "world");
```

#### `print!`

This and `println` emit their output to stdout. Similarly to the `write!` macro,
the goal of these macros is to avoid intermediate allocations when printing
output. Example usage is:

```rust
print!("Hello {}!", "world");
println!("I have a newline {}", "character at the end");
```

#### `format_args!`
This is a curious macro which is used to safely pass around
an opaque object describing the format string. This object
does not require any heap allocations to create, and it only
references information on the stack. Under the hood, all of
the related macros are implemented in terms of this. First
off, some example usage is:

```rust,ignore
use std::fmt;

# fn lol<T>() -> T { fail!() }
# let my_writer: &mut ::std::io::Writer = lol();
# let my_fn: fn(&fmt::Arguments) = lol();

format_args!(fmt::format, "this returns {}", "~str");
format_args!(|args| { fmt::write(my_writer, args) }, "some {}", "args");
format_args!(my_fn, "format {}", "string");
```

The first argument of the `format_args!` macro is a function (or closure) which
takes one argument of type `&fmt::Arguments`. This structure can then be
passed to the `write` and `format` functions inside this module in order to
process the format string. The goal of this macro is to even further prevent
intermediate allocations when dealing formatting strings.

For example, a logging library could use the standard formatting syntax, but it
would internally pass around this structure until it has been determined where
output should go to.

It is unsafe to programmatically create an instance of `fmt::Arguments` because
the operations performed when executing a format string require the compile-time
checks provided by the compiler. The `format_args!` macro is the only method of
safely creating these structures, but they can be unsafely created with the
constructor provided.

## Internationalization

The formatting syntax supported by the `format!` extension supports
internationalization by providing "methods" which execute various different
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

```rust
format!("{0, select, other{#}}", "hello"); // => ~"hello"
```

This example is the equivalent of `{0:s}` essentially.

### Select

The select method is a switch over a `&str` parameter, and the parameter *must*
be of the type `&str`. An example of the syntax is:

```ignore
{0, select, male{...} female{...} other{...}}
```

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

```ignore
{0, plural, offset=1 =1{...} two{...} many{...} other{...}}
```

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

```ignore
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
```

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

## Escaping

The literal characters `{`, `}`, or `#` may be included in a string by
preceding them with the `\` character. Since `\` is already an
escape character in Rust strings, a string literal using this escape
will look like `"\\{"`.

*/

use any;
use cast;
use char::Char;
use container::Container;
use io::MemWriter;
use io;
use iter::{Iterator, range};
use num::Signed;
use option::{Option,Some,None};
use repr;
use result::{Ok, Err};
use str::StrSlice;
use str;
use vec::ImmutableVector;
use vec;

pub mod parse;
pub mod rt;

pub type Result = io::IoResult<()>;

/// A struct to represent both where to emit formatting strings to and how they
/// should be formatted. A mutable version of this is passed to all formatting
/// traits.
pub struct Formatter<'a> {
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
    buf: &'a mut io::Writer,
    priv curarg: vec::Items<'a, Argument<'a>>,
    priv args: &'a [Argument<'a>],
}

/// This struct represents the generic "argument" which is taken by the Xprintf
/// family of functions. It contains a function to format the given value. At
/// compile time it is ensured that the function and the value have the correct
/// types, and then this struct is used to canonicalize arguments to one type.
pub struct Argument<'a> {
    priv formatter: extern "Rust" fn(&any::Void, &mut Formatter) -> Result,
    priv value: &'a any::Void,
}

impl<'a> Arguments<'a> {
    /// When using the format_args!() macro, this function is used to generate the
    /// Arguments structure. The compiler inserts an `unsafe` block to call this,
    /// which is valid because the compiler performs all necessary validation to
    /// ensure that the resulting call to format/write would be safe.
    #[doc(hidden)] #[inline]
    pub unsafe fn new<'a>(fmt: &'static [rt::Piece<'static>],
                          args: &'a [Argument<'a>]) -> Arguments<'a> {
        Arguments{ fmt: cast::transmute(fmt), args: args }
    }
}

/// This structure represents a safely precompiled version of a format string
/// and its arguments. This cannot be generated at runtime because it cannot
/// safely be done so, so no constructors are given and the fields are private
/// to prevent modification.
///
/// The `format_args!` macro will safely create an instance of this structure
/// and pass it to a user-supplied function. The macro validates the format
/// string at compile-time so usage of the `write` and `format` functions can
/// be safely performed.
pub struct Arguments<'a> {
    priv fmt: &'a [rt::Piece<'a>],
    priv args: &'a [Argument<'a>],
}

/// When a format is not otherwise specified, types are formatted by ascribing
/// to this trait. There is not an explicit way of selecting this trait to be
/// used for formatting, it is only if no other format is specified.
#[allow(missing_doc)]
pub trait Show { fn fmt(&self, &mut Formatter) -> Result; }

/// Format trait for the `b` character
#[allow(missing_doc)]
pub trait Bool { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `c` character
#[allow(missing_doc)]
pub trait Char { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `i` and `d` characters
#[allow(missing_doc)]
pub trait Signed { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `u` character
#[allow(missing_doc)]
pub trait Unsigned { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `o` character
#[allow(missing_doc)]
pub trait Octal { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `b` character
#[allow(missing_doc)]
pub trait Binary { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `x` character
#[allow(missing_doc)]
pub trait LowerHex { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `X` character
#[allow(missing_doc)]
pub trait UpperHex { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `s` character
#[allow(missing_doc)]
pub trait String { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `?` character
#[allow(missing_doc)]
pub trait Poly { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `p` character
#[allow(missing_doc)]
pub trait Pointer { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `f` character
#[allow(missing_doc)]
pub trait Float { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `e` character
#[allow(missing_doc)]
pub trait LowerExp { fn fmt(&self, &mut Formatter) -> Result; }
/// Format trait for the `E` character
#[allow(missing_doc)]
pub trait UpperExp { fn fmt(&self, &mut Formatter) -> Result; }

// FIXME #11938 - UFCS would make us able call the above methods
// directly Show::show(x, fmt).

// FIXME(huonw's WIP): this is a intermediate state waiting for a
// snapshot (at the time of writing we're at 2014-01-20 b6400f9), to
// be able to make the `fmt` functions into normal methods and have
// `format!()` still work.
macro_rules! uniform_fn_call_workaround {
    ($( $name: ident, $trait_: ident; )*) => {
        $(
            #[doc(hidden)]
            pub fn $name<T: $trait_>(x: &T, fmt: &mut Formatter) -> Result {
                x.fmt(fmt)
            }
            )*
    }
}
uniform_fn_call_workaround! {
    secret_show, Show;
    secret_bool, Bool;
    secret_char, Char;
    secret_signed, Signed;
    secret_unsigned, Unsigned;
    secret_octal, Octal;
    secret_binary, Binary;
    secret_lower_hex, LowerHex;
    secret_upper_hex, UpperHex;
    secret_string, String;
    secret_poly, Poly;
    secret_pointer, Pointer;
    secret_float, Float;
    secret_lower_exp, LowerExp;
    secret_upper_exp, UpperExp;
}

/// The `write` function takes an output stream, a precompiled format string,
/// and a list of arguments. The arguments will be formatted according to the
/// specified format string into the output stream provided.
///
/// # Arguments
///
///   * output - the buffer to write output to
///   * args - the precompiled arguments generated by `format_args!`
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::fmt;
/// use std::io;
///
/// let w = &mut io::stdout() as &mut io::Writer;
/// format_args!(|args| { fmt::write(w, args); }, "Hello, {}!", "world");
/// ```
pub fn write(output: &mut io::Writer, args: &Arguments) -> Result {
    unsafe { write_unsafe(output, args.fmt, args.args) }
}

/// The `writeln` function takes the same arguments as `write`, except that it
/// will also write a newline (`\n`) character at the end of the format string.
pub fn writeln(output: &mut io::Writer, args: &Arguments) -> Result {
    let first = unsafe { write_unsafe(output, args.fmt, args.args) };
    first.and_then(|()| output.write(['\n' as u8]))
}

/// The `write_unsafe` function takes an output stream, a precompiled format
/// string, and a list of arguments. The arguments will be formatted according
/// to the specified format string into the output stream provided.
///
/// See the documentation for `format` for why this function is unsafe and care
/// should be taken if calling it manually.
///
/// Thankfully the rust compiler provides macros like `write!` and
/// `format_args!` which perform all of this validation at compile-time
/// and provide a safe interface for invoking this function.
///
/// # Arguments
///
///   * output - the buffer to write output to
///   * fmts - the precompiled format string to emit
///   * args - the list of arguments to the format string. These are only the
///            positional arguments (not named)
///
/// Note that this function assumes that there are enough arguments for the
/// format string.
pub unsafe fn write_unsafe(output: &mut io::Writer,
                           fmt: &[rt::Piece],
                           args: &[Argument]) -> Result {
    let mut formatter = Formatter {
        flags: 0,
        width: None,
        precision: None,
        buf: output,
        align: parse::AlignUnknown,
        fill: ' ',
        args: args,
        curarg: args.iter(),
    };
    for piece in fmt.iter() {
        if_ok!(formatter.run(piece, None));
    }
    Ok(())
}

/// The format function takes a precompiled format string and a list of
/// arguments, to return the resulting formatted string.
///
/// # Arguments
///
///   * args - a structure of arguments generated via the `format_args!` macro.
///            Because this structure can only be safely generated at
///            compile-time, this function is safe.
///
/// # Example
///
/// ```rust
/// use std::fmt;
///
/// let s = format_args!(fmt::format, "Hello, {}!", "world");
/// assert_eq!(s, ~"Hello, world!");
/// ```
pub fn format(args: &Arguments) -> ~str {
    unsafe { format_unsafe(args.fmt, args.args) }
}

/// The unsafe version of the formatting function.
///
/// This is currently an unsafe function because the types of all arguments
/// aren't verified by immediate callers of this function. This currently does
/// not validate that the correct types of arguments are specified for each
/// format specifier, nor that each argument itself contains the right function
/// for formatting the right type value. Because of this, the function is marked
/// as `unsafe` if this is being called manually.
///
/// Thankfully the rust compiler provides the macro `format!` which will perform
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
pub unsafe fn format_unsafe(fmt: &[rt::Piece], args: &[Argument]) -> ~str {
    let mut output = MemWriter::new();
    write_unsafe(&mut output as &mut io::Writer, fmt, args).unwrap();
    return str::from_utf8_owned(output.unwrap()).unwrap();
}

impl<'a> Formatter<'a> {

    // First up is the collection of functions used to execute a format string
    // at runtime. This consumes all of the compile-time statics generated by
    // the format! syntax extension.

    fn run(&mut self, piece: &rt::Piece, cur: Option<&str>) -> Result {
        match *piece {
            rt::String(s) => self.buf.write(s.as_bytes()),
            rt::CurrentArgument(()) => self.buf.write(cur.unwrap().as_bytes()),
            rt::Argument(ref arg) => {
                // Fill in the format parameters into the formatter
                self.fill = arg.format.fill;
                self.align = arg.format.align;
                self.flags = arg.format.flags;
                self.width = self.getcount(&arg.format.width);
                self.precision = self.getcount(&arg.format.precision);

                // Extract the correct argument
                let value = match arg.position {
                    rt::ArgumentNext => { *self.curarg.next().unwrap() }
                    rt::ArgumentIs(i) => self.args[i],
                };

                // Then actually do some printing
                match arg.method {
                    None => (value.formatter)(value.value, self),
                    Some(ref method) => self.execute(*method, value)
                }
            }
        }
    }

    fn getcount(&mut self, cnt: &rt::Count) -> Option<uint> {
        match *cnt {
            rt::CountIs(n) => { Some(n) }
            rt::CountImplied => { None }
            rt::CountIsParam(i) => {
                let v = self.args[i].value;
                unsafe { Some(*(v as *any::Void as *uint)) }
            }
            rt::CountIsNextParam => {
                let v = self.curarg.next().unwrap().value;
                unsafe { Some(*(v as *any::Void as *uint)) }
            }
        }
    }

    fn execute(&mut self, method: &rt::Method, arg: Argument) -> Result {
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
                        rt::Literal(val) if value == val => {
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
                        rt::Keyword(parse::Zero) => value == 0,
                        rt::Keyword(parse::One) => value == 1,
                        rt::Keyword(parse::Two) => value == 2,

                        // FIXME: Few/Many should have a user-specified boundary
                        //      One possible option would be in the function
                        //      pointer of the 'arg: Argument' struct.
                        rt::Keyword(parse::Few) => value < 8,
                        rt::Keyword(parse::Many) => value >= 8,

                        rt::Literal(..) => false
                    };
                    if run {
                        return self.runplural(value, s.result);
                    }
                }

                self.runplural(value, *default)
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
                            if_ok!(self.run(piece, Some(value)));
                        }
                        return Ok(());
                    }
                }
                for piece in default.iter() {
                    if_ok!(self.run(piece, Some(value)));
                }
                Ok(())
            }
        }
    }

    fn runplural(&mut self, value: uint, pieces: &[rt::Piece]) -> Result {
        ::uint::to_str_bytes(value, 10, |buf| {
            let valuestr = str::from_utf8(buf).unwrap();
            for piece in pieces.iter() {
                if_ok!(self.run(piece, Some(valuestr)));
            }
            Ok(())
        })
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
                        positive: bool) -> Result {
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
                    if_ok!(this.buf.write(['+' as u8]));
                } else if !positive {
                    if_ok!(this.buf.write(['-' as u8]));
                }
                if this.flags & 1 << (FlagAlternate as uint) != 0 {
                    if_ok!(this.buf.write(alternate_prefix.as_bytes()));
                }
                signprinted = true;
            }
            Ok(())
        };

        let emit = |this: &mut Formatter| {
            sign(this).and_then(|()| this.buf.write(s))
        };

        match self.width {
            None => emit(self),
            Some(min) if actual_len >= min => emit(self),
            Some(min) => {
                if self.flags & 1 << (FlagSignAwareZeroPad as uint) != 0 {
                    self.fill = '0';
                    if_ok!(sign(self));
                }
                self.with_padding(min - actual_len, parse::AlignRight, |me| {
                    emit(me)
                })
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
    pub fn pad(&mut self, s: &str) -> Result {
        // Make sure there's a fast path up front
        if self.width.is_none() && self.precision.is_none() {
            return self.buf.write(s.as_bytes());
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
                    let nchars = ::cmp::min(max, char_len);
                    return self.buf.write(s.slice_chars(0, nchars).as_bytes());
                }
            }
            None => {}
        }

        // The `width` field is more of a `min-width` parameter at this point.
        match self.width {
            // If we're under the maximum length, and there's no minimum length
            // requirements, then we can just emit the string
            None => self.buf.write(s.as_bytes()),

            // If we're under the maximum width, check if we're over the minimum
            // width, if so it's as easy as just emitting the string.
            Some(width) if s.char_len() >= width => {
                self.buf.write(s.as_bytes())
            }

            // If we're under both the maximum and the minimum width, then fill
            // up the minimum width with the specified string + some alignment.
            Some(width) => {
                self.with_padding(width - s.len(), parse::AlignLeft, |me| {
                    me.buf.write(s.as_bytes())
                })
            }
        }
    }

    fn with_padding(&mut self,
                    padding: uint,
                    default: parse::Alignment,
                    f: |&mut Formatter| -> Result) -> Result {
        let align = match self.align {
            parse::AlignUnknown => default,
            parse::AlignLeft | parse::AlignRight => self.align
        };
        if align == parse::AlignLeft {
            if_ok!(f(self));
        }
        let mut fill = [0u8, ..4];
        let len = self.fill.encode_utf8(fill);
        for _ in range(0, padding) {
            if_ok!(self.buf.write(fill.slice_to(len)));
        }
        if align == parse::AlignRight {
            if_ok!(f(self));
        }
        Ok(())
    }
}

/// This is a function which calls are emitted to by the compiler itself to
/// create the Argument structures that are passed into the `format` function.
#[doc(hidden)] #[inline]
pub fn argument<'a, T>(f: extern "Rust" fn(&T, &mut Formatter) -> Result,
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
#[doc(hidden)] #[inline]
pub fn argumentstr<'a>(s: &'a &str) -> Argument<'a> {
    argument(secret_string, s)
}

/// When the compiler determines that the type of an argument *must* be a uint
/// (such as for plural), then it invokes this method.
#[doc(hidden)] #[inline]
pub fn argumentuint<'a>(s: &'a uint) -> Argument<'a> {
    argument(secret_unsigned, s)
}

// Implementations of the core formatting traits

impl Bool for bool {
    fn fmt(&self, f: &mut Formatter) -> Result {
        secret_string(&(if *self {"true"} else {"false"}), f)
    }
}

impl<'a, T: str::Str> String for T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.pad(self.as_slice())
    }
}

impl Char for char {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut utf8 = [0u8, ..4];
        let amt = self.encode_utf8(utf8);
        let s: &str = unsafe { cast::transmute(utf8.slice_to(amt)) };
        secret_string(&s, f)
    }
}

macro_rules! int_base(($ty:ident, $into:ident, $base:expr,
                       $name:ident, $prefix:expr) => {
    impl $name for $ty {
        fn fmt(&self, f: &mut Formatter) -> Result {
            ::$into::to_str_bytes(*self as $into, $base, |buf| {
                f.pad_integral(buf, $prefix, true)
            })
        }
    }
})
macro_rules! upper_hex(($ty:ident, $into:ident) => {
    impl UpperHex for $ty {
        fn fmt(&self, f: &mut Formatter) -> Result {
            ::$into::to_str_bytes(*self as $into, 16, |buf| {
                upperhex(buf, f)
            })
        }
    }
})
// Not sure why, but this causes an "unresolved enum variant, struct or const"
// when inlined into the above macro...
#[doc(hidden)]
pub fn upperhex(buf: &[u8], f: &mut Formatter) -> Result {
    let mut local = [0u8, ..16];
    for i in ::iter::range(0, buf.len()) {
        local[i] = match buf[i] as char {
            'a' .. 'f' => (buf[i] - 'a' as u8) + 'A' as u8,
            c => c as u8,
        }
    }
    f.pad_integral(local.slice_to(buf.len()), "0x", true)
}

macro_rules! integer(($signed:ident, $unsigned:ident) => {
    // Signed is special because it actuall emits the negative sign,
    // nothing else should do that, however.
    impl Signed for $signed {
        fn fmt(&self, f: &mut Formatter) -> Result {
            ::$unsigned::to_str_bytes(self.abs() as $unsigned, 10, |buf| {
                f.pad_integral(buf, "", *self >= 0)
            })
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
})

integer!(int, uint)
integer!(i8, u8)
integer!(i16, u16)
integer!(i32, u32)
integer!(i64, u64)

macro_rules! floating(($ty:ident) => {
    impl Float for $ty {
        fn fmt(&self, fmt: &mut Formatter) -> Result {
            // FIXME: this shouldn't perform an allocation
            let s = match fmt.precision {
                Some(i) => ::$ty::to_str_exact(self.abs(), i),
                None => ::$ty::to_str_digits(self.abs(), 6)
            };
            fmt.pad_integral(s.as_bytes(), "", *self >= 0.0)
        }
    }

    impl LowerExp for $ty {
        fn fmt(&self, fmt: &mut Formatter) -> Result {
            // FIXME: this shouldn't perform an allocation
            let s = match fmt.precision {
                Some(i) => ::$ty::to_str_exp_exact(self.abs(), i, false),
                None => ::$ty::to_str_exp_digits(self.abs(), 6, false)
            };
            fmt.pad_integral(s.as_bytes(), "", *self >= 0.0)
        }
    }

    impl UpperExp for $ty {
        fn fmt(&self, fmt: &mut Formatter) -> Result {
            // FIXME: this shouldn't perform an allocation
            let s = match fmt.precision {
                Some(i) => ::$ty::to_str_exp_exact(self.abs(), i, true),
                None => ::$ty::to_str_exp_digits(self.abs(), 6, true)
            };
            fmt.pad_integral(s.as_bytes(), "", *self >= 0.0)
        }
    }
})
floating!(f32)
floating!(f64)

impl<T> Poly for T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match (f.width, f.precision) {
            (None, None) => {
                repr::write_repr(f.buf, self)
            }

            // If we have a specified width for formatting, then we have to make
            // this allocation of a new string
            _ => {
                let s = repr::repr_to_str(self);
                f.pad(s)
            }
        }
    }
}

impl<T> Pointer for *T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.flags |= 1 << (parse::FlagAlternate as uint);
        ::uint::to_str_bytes(*self as uint, 16, |buf| {
            f.pad_integral(buf, "0x", true)
        })
    }
}
impl<T> Pointer for *mut T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        secret_pointer::<*T>(&(*self as *T), f)
    }
}
impl<'a, T> Pointer for &'a T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        secret_pointer::<*T>(&(&**self as *T), f)
    }
}
impl<'a, T> Pointer for &'a mut T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        secret_pointer::<*T>(&(&**self as *T), f)
    }
}

// Implementation of Show for various core types

macro_rules! delegate(($ty:ty to $other:ident) => {
    impl<'a> Show for $ty {
        fn fmt(&self, f: &mut Formatter) -> Result {
            (concat_idents!(secret_, $other)(self, f))
        }
    }
})
delegate!(int to signed)
delegate!( i8 to signed)
delegate!(i16 to signed)
delegate!(i32 to signed)
delegate!(i64 to signed)
delegate!(uint to unsigned)
delegate!(  u8 to unsigned)
delegate!( u16 to unsigned)
delegate!( u32 to unsigned)
delegate!( u64 to unsigned)
delegate!(~str to string)
delegate!(&'a str to string)
delegate!(bool to bool)
delegate!(char to char)
delegate!(f32 to float)
delegate!(f64 to float)

impl<T> Show for *T {
    fn fmt(&self, f: &mut Formatter) -> Result { secret_pointer(self, f) }
}
impl<T> Show for *mut T {
    fn fmt(&self, f: &mut Formatter) -> Result { secret_pointer(self, f) }
}

// If you expected tests to be here, look instead at the run-pass/ifmt.rs test,
// it's a lot easier than creating all of the rt::Piece structures here.
