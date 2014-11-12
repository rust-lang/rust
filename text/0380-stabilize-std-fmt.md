- Start Date: 2014-11-12
- RFC PR: [rust-lang/rfcs#380](https://github.com/rust-lang/rfcs/pull/380)
- Rust Issue: [rust-lang/rust#18904](https://github.com/rust-lang/rust/issues/18904)

# Summary

Stabilize the `std::fmt` module, in addition to the related macros and
formatting language syntax. As a high-level summary:

* Leave the format syntax as-is.
* Remove a number of superfluous formatting traits (renaming a few in the
  process).

# Motivation

This RFC is primarily motivated by the need to stabilize `std::fmt`. In the past
stabilization has not required RFCs, but the changes envisioned for this module
are far-reaching and modify some parts of the language (format syntax), leading
to the conclusion that this stabilization effort required an RFC.

# Detailed design

The `std::fmt` module encompasses more than just the actual
structs/traits/functions/etc defined within it, but also a number of macros and
the formatting language syntax for describing format strings. Each of these
features of the module will be described in turn.

## Formatting Language Syntax

The [documented syntax](http://doc.rust-lang.org/std/fmt/#syntax) will not be
changing as-written. All of these features will be accepted wholesale
(considered stable):

* Usage of `{}` for "format something here" placeholders
* `{{` as an escape for `{` (and vice-versa for `}`)
* Various format specifiers
  * fill character for alignment
  * actual alignment, left (`<`), center (`^`), and right (`>`).
  * sign to print (`+` or `-`)
  * minimum width for text to be printed
    * both a literal count and a runtime argument to the format string
  * precision or maximum width
    * all of a literal count, a specific runtime argument to the format string,
      and "the next" runtime argument to the format string.
  * "alternate formatting" (`#`)
  * leading zeroes (`0`)
* Integer specifiers of what to format (`{0}`)
* Named arguments (`{foo}`)

### Using Format Specifiers

While quite useful occasionally, there is no static guarantee that any
implementation of a formatting trait actually respects the format specifiers
passed in. For example, this code does not necessarily work as expected:

```rust
#[deriving(Show)]
struct A;

format!("{:10}", A);
```

All of the primitives for rust (strings, integers, etc) have implementations of
`Show` which respect these formatting flags, but almost no other implementations
do (notably those generated via `deriving`).

This RFC proposes stabilizing the formatting flags, despite this current state
of affairs. There are in theory possible alternatives in which there is a
static guarantee that a type does indeed respect format specifiers when one is
provided, generating a compile-time error when a type doesn't respect a
specifier. These alternatives, however, appear to be too heavyweight and are
considered somewhat overkill.

In general it's trivial to respect format specifiers if an implementation
delegates to a primitive or somehow has a buffer of what's to be formatted. To
cover these two use cases, the `Formatter` structure passed around has helper
methods to assist in formatting these situations. This is, however, quite rare
to fall into one of these two buckets, so the specifiers are largely ignored
(and the formatter is `write!`-n to directly).

### Named Arguments

Currently Rust does not support named arguments anywhere *except* for format
strings. Format strings can get away with it because they're all part of a macro
invocation (unlike the rest of Rust syntax).

The worry for stabilizing a named argument syntax for the formatting language is
that if Rust ever adopts named arguments with a *different* syntax, it would be
quite odd having two systems.

The most recently proposed [keyword argument
RFC](https://github.com/rust-lang/rfcs/pull/257) used `:` for the invocation
syntax rather than `=` as formatting does today. Additionally, today `foo = bar`
is a valid expression, having a value of type `()`.

With these worries, there are one of two routes that could be pursued:

1. The `expr = expr` syntax could be disallowed on the language level. This
   could happen both in a total fashion or just allowing the expression
   appearing as a function argument. For both cases, this will probably be
   considered a "wart" of Rust's grammar.
2. The `foo = bar` syntax could be allowed in the macro with prior knowledge
   that the default argument syntax for Rust, if one is ever developed, will
   likely be different. This would mean that the `foo = bar` syntax in
   formatting macros will likely be considered a wart in the future.

Given these two cases, the clear choice seems to be accepting a wart in the
formatting macros themselves. It will likely be possible to extend the macro in
the future to support whatever named argument syntax is developed as well, and
the old syntax could be accepted for some time.

## Formatting Traits

Today there are 16 formatting traits. Each trait represents a "type" of
formatting, corresponding to the `[type]` production in the formatting syntax.
As a bit of history, the original intent was for each trait to declare what
specifier it used, allowing users to add more specifiers in newer crates. For
example the `time` crate could provide the `{:time}` formatting trait. This
design was seen as too complicated, however, so it was not landed. It does,
however, partly motivate why there is one trait per format specifier today.

The 16 formatting traits and their format specifiers are:

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

This RFC proposes removing the following traits:

* `Signed`
* `Unsigned`
* `Bool`
* `Char`
* `String`
* `Float`

Note that this RFC would like to remove `Poly`, but that is covered by [a
separate RFC](https://github.com/rust-lang/rfcs/pull/379).

Today by far the most common formatting trait is `Show`, and over time the
usefulness of these formatting traits has been reduced. The traits this RFC
proposes to remove are only assertions that the type provided actually
implements the trait, there are few known implementations of the traits which
diverge on how they are implemented.

Additionally, there are a two of oddities inherited from ancient C:

* Both `d` and `i` are wired to `Signed`
* One may reasonable expect the `Binary` trait to use `b` as its specifier.

The remaining traits this RFC recommends leaving. The rationale for this is that
they represent alternate representations of primitive types in general, and are
also quite often expected when coming from other format syntaxes such as
C/Python/Ruby/etc.

It would, of course, be possible to re-add any of these traits in a
backwards-compatible fashion.

### Format type for `Binary`

With the removal of the `Bool` trait, this RFC recommends renaming the specifier
for `Binary` to `b` instead of `t`.

### Combining all traits

A possible alternative to having many traits is to instead have one trait, such
as:

```rust
pub trait Show {
    fn fmt(...);
    fn hex(...) { fmt(...) }
    fn lower_hex(...) { fmt(...) }
    ...
}
```

There are a number of pros to this design:

* Instead of having to consider many traits, only one trait needs to be
  considered.
* All types automatically implement all format types or zero format types.
* In a hypothetical world where a format string could be constructed at runtime,
  this would alleviate the signature of such a function. The concrete type taken
  for all its arguments would be `&Show` and then if the format string supplied
  `:x` or `:o` the runtime would simply delegate to the relevant trait method.

There are also a number of cons to this design, which motivate this RFC
recommending the remaining separation of these traits.

* The "static assertion" that a type implements a relevant format trait becomes
  almost nonexistent because all types either implement none or all formatting
  traits.
* The documentation for the `Show` trait becomes somewhat overwhelming because
  it's no longer immediately clear which method should be overridden for what.
* A hypothetical world with runtime format string construction could find a
  different system for taking arguments.

### Method signature

Currently, each formatting trait has a signature as follows:

```rust
fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
```

This implies that all formatting is considered to be a stream-oriented operation
where `f` is a sink to write bytes to. The `fmt::Result` type indicates that
some form of "write error" happened, but conveys no extra information.

This API has a number of oddities:

* The type `Formatter` has inherent `write` and `write_fmt` methods to be used
  in conjuction with the `write!` macro return an instance of `fmt::Result`.
* The `Formatter` type also implements the `std::io::Writer` trait in order to
  be able to pass around a `&mut Writer`.
* This relies on the duck-typing of macros and for the inherent `write_fmt`
  method to trump the `Writer`'s `write_fmt` method in order to return an error
  of the correct type.
* The `Result` return type is an enumeration with precisely one variant,
  `FormatError`.

Overall, this signature seems to be appropriate in terms of "give me a sink of
bytes to write myself to, and let me return an error if one happens". Due to
this, this RFC recommends that all formatting traits be marked `#[unstable]`.

## Macros

There are a number of prelude macros which interact with the format syntax:

* `format_args`
* `format_args_method`
* `write`
* `writeln`
* `print`
* `println`
* `format`
* `fail`
* `assert`
* `debug_assert`

All of these are `macro_rules!`-defined macros, except for `format_args` and
`format_args_method`.

### Common syntax

All of these macros take some form of prefix, while the trailing suffix is
always some instantiation of the formatting syntax. The suffix portion is
recommended to be considered `#[stable]`, and the sections below will discuss
each macro in detail with respect to its prefix and semantics.

### format_args

The fundamental purpose of this macro is to generate a value of type
`&fmt::Arguments` which represents a pending format computation. This structure
can then be passed at some point to the methods in `std::fmt` to actually
perform the format.

The prefix of this macro is some "callable thing", be it a top-level function or
a closure. It cannot invoke a method because `foo.bar` is not a "callable thing"
to call the `bar` method on `foo`.

Ideally, this macro would have no prefix, and would be callable like:

```rust
use std::fmt;

let args = format_args!("Hello {}!", "world");
let hello_world = fmt::format(args);
```

Unfortunately, without an implementation of [RFC 31][rfc-31] this is not
possible. As a result, this RFC proposes a `#[stable]` consideration of this
macro and its syntax.

[rfc-31]: https://github.com/rust-lang/rfcs/blob/master/active/0031-better-temporary-lifetimes.md

### format_args_method

The purpose of this macro is to solve the "call this method" case not covered
with the `format_args` macro. This macro was introduced fairly late in the game
to solve the problem that `&*trait_object` was not allowed. This is currently
allowed, however (due to DST).

This RFC proposes immediately removing this macro. The primary user of this
macro is `write!`, meaning that the following code, which compiles today, would
need to be rewritten:

```rust
let mut output = std::io::stdout();
// note the lack of `&mut` in front
write!(output, "hello {}", "world");
```

The `write!` macro would be redefined as:

```rust
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => ({
        let dst = &mut *$dst;
        format_args!(|args| { dst.write_fmt(args) }, $($arg)*)
    })
)
```

The purpose here is to borrow `$dst` *outside* of the closure to ensure that the
closure doesn't borrow too many of its contents. Otherwise, code such as this
would be disallowed

```rust
write!(&mut my_struct.writer, "{}", my_struct.some_other_field);
```

### write/writeln

These two macros take the prefix of "some pointer to a writer" as an argument,
and then format data into the write (returning whatever `write_fmt` returns).
These macros were originally designed to require a `&mut T` as the first
argument, but today, due to the usage of `format_args_method`, they can take any
`T` which responds to `write_fmt`.

This RFC recommends marking these two macros `#[stable]` with the modification
above (removing `format_args_method`). The `ln` suffix to `writeln` will be
discussed shortly.

### print/println

These two macros take no prefix, and semantically print to a *task-local* stdout
stream. The purpose of a task-local stream is provide some form of buffering to
make stdout printing at all performant.

This RFC recommends marking these two macros a `#[stable]`.

#### The `ln` suffix

The name `println` is one of the few locations in Rust where a short C-like
abbreviation is accepted rather than the more verbose, but clear, `print_line`
(for example). Due to the overwhelming precedent of other languages (even Java
uses `println`!), this is seen as an acceptable special case to the rule.

### format

This macro takes no prefix and returns a `String`.

In ancient rust this macro was called its shorter name, `fmt`. Additionally, the
name `format` is somewhat inconsistent with the module name of `fmt`. Despite
this, this RFC recommends considering this macro `#[stable]` due to its
delegation to the `format` method in the `std::fmt` module, similar to how the
`write!` macro delegates to the `fmt::write`.

### fail/assert/debug_assert

The format string portions of these macros are recommended to be considered as
`#[stable]` as part of this RFC. The actual stability of the macros is not
considered as part of this RFC.

## Freestanding Functions

There are a number of [freestanding
functions](http://doc.rust-lang.org/std/fmt/index.html#functions) to consider in
the `std::fmt` module for stabilization.

* `fn format(args: &Arguments) -> String`

  This RFC recommends `#[experimental]`. This method is largely an
  implementation detail of this module, and should instead be used via:

  ```rust
  let args: &fmt::Arguments = ...;
  format!("{}", args)
  ```

* `fn write(output: &mut FormatWriter, args: &Arguments) -> Result`

  This is somewhat surprising in that the argument to this function is not a
  `Writer`, but rather a `FormatWriter`. This is technically speaking due to the
  core/std separation and how this function is defined in core and `Writer` is
  defined in std.

  This RFC recommends marking this function `#[experimental]` as the
  `write_fmt` exists on `Writer` to perform the corresponding operation.
  Consequently we may wish to remove this function in favor of the `write_fmt`
  method on `FormatWriter`.

  Ideally this method would be removed from the public API as it is just an
  implementation detail of the `write!` macro.

* `fn radix<T>(x: T, base: u8) -> RadixFmt<T, Radix>`

  This function is a bit of an odd-man-out in that it is a constructor, but does
  not follow the existing conventions of `Type::new`. The purpose of this
  function is to expose the ability to format a number for any radix. The
  default format specifiers `:o`, `:x`, and `:t` are essentially shorthands for
  this function, except that the format types have specialized implementations
  per radix instead of a generic implementation.

  This RFC proposes that this function be considered `#[unstable]` as its
  location and naming are a bit questionable, but the functionality is desired.

## Miscellaneous items

* `trait FormatWriter`

  This trait is currently the actual implementation strategy of formatting, and
  is defined specially in libcore. It is rarely used outside of libcore. It is
  recommended to be `#[experimental]`.

  There are possibilities in moving `Reader` and `Writer` to libcore with the
  error type as an associated item, allowing the `FormatWriter` trait to be
  eliminated entirely. Due to this possibility, the trait will be experimental
  for now as alternative solutions are explored.

* `struct Argument`, `mod rt`, `fn argument`, `fn argumentstr`,
   `fn argumentuint`, `Arguments::with_placeholders`, `Arguments::new`

  These are implementation details of the `Arguments` structure as well as the
  expansion of the `format_args!` macro. It's recommended to mark these as
  `#[experimental]` and `#[doc(hidden)]`. Ideally there would be some form of
  macro-based privacy hygiene which would allow these to be truly private, but
  it will likely be the case that these simply become stable and we must live
  with them forever.

* `struct Arguments`

  This is a representation of a "pending format string" which can be used to
  safely execute a `Formatter` over it. This RFC recommends `#[stable]`.

* `struct Formatter`

  This instance is passed to all formatting trait methods and contains helper
  methods for respecting formatting flags. This RFC recommends `#[unstable]`.

  This RFC also recommends deprecating all public fields in favor of accessor
  methods. This should help provide future extensibility as well as preventing
  unnecessary mutation in the future.

* `enum FormatError`

  This enumeration only has one instance, `WriteError`. It is recommended to
  make this a `struct` instead and rename it to just `Error`. The purpose of
  this is to signal that an error has occurred as part of formatting, but it
  does not provide a generic method to transmit any other information other than
  "an error happened" to maintain the ergonomics of today's usage. It's strongly
  recommended that implementations of `Show` and friends are infallible and only
  generate an error if the underlying `Formatter` returns an error itself.

* `Radix`/`RadixFmt`

  Like the `radix` function, this RFC recommends `#[unstable]` for both of these
  pieces of functionality.

# Drawbacks

Today's macro system necessitates exporting many implementation details of the
formatting system, which is unfortunate.

# Alternatives

A number of alternatives were laid out in the detailed description for various
aspects.

# Unresolved questions

* How feasible and/or important is it to construct a format string at runtime
  given the recommend stability levels in this RFC?
