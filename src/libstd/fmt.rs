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
revision, the `format!` macro returns a `String` type which is the result of
the formatting. In the future it will also be able to pass in a stream to
format arguments directly while performing minimal allocations.

Some examples of the `format!` extension are:

```rust
# extern crate debug;
# fn main() {
format!("Hello");                 // => "Hello"
format!("Hello, {:s}!", "world"); // => "Hello, world!"
format!("The number is {:d}", 1); // => "The number is 1"
format!("{:?}", (3, 4));          // => "(3, 4)"
format!("{value}", value=4);      // => "4"
format!("{} {}", 1, 2);           // => "1 2"
# }
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
format!("{1} {} {0} {}", 1, 2); // => "2 1 1 2"
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

```text
identifier '=' expression
```

For example, the following `format!` expressions all use named argument:

```rust
# extern crate debug;
# fn main() {
format!("{argument}", argument = "test");       // => "test"
format!("{name} {}", 1, name = 2);              // => "2 1"
format!("{a:s} {c:d} {b:?}", a="a", b=(), c=3); // => "a 3 ()"
# }
```

It is illegal to put positional parameters (those without names) after arguments
which have names. Like positional parameters, it is illegal to provided named
parameters that are unused by the format string.

### Argument types

Each argument's type is dictated by the format string. It is a requirement that
every argument is only ever referred to by one type. For example, this is an
invalid format string:

```text
{0:d} {0:s}
```

This is invalid because the first argument is both referred to as an integer as
well as a string.

Because formatting is done via traits, there is no requirement that the
`d` format actually takes an `int`, but rather it simply requires a type which
ascribes to the `Signed` formatting trait. There are various parameters which do
require a particular type, however. Namely if the syntax `{:.*s}` is used, then
the number of characters to print from the string precedes the actual string and
must have the type `uint`. Although a `uint` can be printed with `{:u}`, it is
illegal to reference an argument as such. For example, this is another invalid
format string:

```text
{:.*s} {0:u}
```

### Formatting traits

When requesting that an argument be formatted with a particular type, you are
actually requesting that an argument ascribes to a particular trait. This allows
multiple actual types to be formatted via `{:d}` (like `i8` as well as `int`).
The current mapping of types to traits is:

* *nothing* ⇒ `Show`
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
* `?` ⇒ `Poly`

> **Note**: The `Poly` formatting trait is provided by [libdebug](../../debug/)
> and is an experimental implementation that should not be relied upon. In order
> to use the `?` modifier, the libdebug crate must be linked against.

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
        // The `f` value implements the `Writer` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        write!(f, "({}, {})", self.x, self.y)
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
        f.pad_integral(true, "", string.as_bytes())
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

```ignore
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
# #![allow(unused_must_use)]
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

```
use std::fmt;
use std::io;

# #[allow(unused_must_use)]
# fn main() {
format_args!(fmt::format, "this returns {}", "String");

let some_writer: &mut io::Writer = &mut io::stdout();
format_args!(|args| { write!(some_writer, "{}", args) }, "print with a {}", "closure");

fn my_fmt_fn(args: &fmt::Arguments) {
    write!(&mut io::stdout(), "{}", args);
}
format_args!(my_fmt_fn, "or a {} too", "function");
# }
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

## Syntax

The syntax for the formatting language used is drawn from other languages, so it
should not be too alien. Arguments are formatted with python-like syntax,
meaning that arguments are surrounded by `{}` instead of the C-like `%`. The
actual grammar for the formatting syntax is:

```text
format_string := <text> [ format <text> ] *
format := '{' [ argument ] [ ':' format_spec ] '}'
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

The literal characters `{` and `}` may be included in a string by preceding them
with the same character. For example, the `{` character is escaped with `{{` and
the `}` character is escaped with `}}`.

*/

use io::Writer;
use io;
use result::{Ok, Err};
use str::{Str, StrAllocating};
use str;
use string;
use slice::Vector;

pub use core::fmt::{Formatter, Result, FormatWriter, rt};
pub use core::fmt::{Show, Bool, Char, Signed, Unsigned, Octal, Binary};
pub use core::fmt::{LowerHex, UpperHex, String, Pointer};
pub use core::fmt::{Float, LowerExp, UpperExp};
pub use core::fmt::{FormatError, WriteError};
pub use core::fmt::{Argument, Arguments, write, radix, Radix, RadixFmt};

#[doc(hidden)]
pub use core::fmt::{argument, argumentstr, argumentuint};
#[doc(hidden)]
pub use core::fmt::{secret_show, secret_string, secret_unsigned};
#[doc(hidden)]
pub use core::fmt::{secret_signed, secret_lower_hex, secret_upper_hex};
#[doc(hidden)]
pub use core::fmt::{secret_bool, secret_char, secret_octal, secret_binary};
#[doc(hidden)]
pub use core::fmt::{secret_bool, secret_char, secret_octal, secret_binary};
#[doc(hidden)]
pub use core::fmt::{secret_float, secret_upper_exp, secret_lower_exp};
#[doc(hidden)]
pub use core::fmt::{secret_pointer};

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
/// assert_eq!(s, "Hello, world!".to_string());
/// ```
pub fn format(args: &Arguments) -> string::String{
    let mut output = io::MemWriter::new();
    let _ = write!(&mut output, "{}", args);
    str::from_utf8(output.unwrap().as_slice()).unwrap().into_string()
}

impl<'a> Writer for Formatter<'a> {
    fn write(&mut self, b: &[u8]) -> io::IoResult<()> {
        match (*self).write(b) {
            Ok(()) => Ok(()),
            Err(WriteError) => Err(io::standard_error(io::OtherIoError))
        }
    }
}
