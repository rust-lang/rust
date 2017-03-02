- Start Date: 2014-12-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/504
- Rust Issue: https://github.com/rust-lang/rust/issues/20013

# Summary

Today's `Show` trait will be tasked with the purpose of providing the ability to
inspect the representation of implementors of the trait. A new trait, `String`,
will be introduced to the `std::fmt` module to in order to represent data that
can essentially be serialized to a string, typically representing the precise
internal state of the implementor.

The `String` trait will take over the `{}` format specifier and the `Show` trait
will move to the now-open `{:?}` specifier.

# Motivation

The formatting traits today largely provide clear guidance to what they are
intended for. For example the `Binary` trait is intended for printing the binary
representation of a data type. The ubiquitous `Show` trait, however, is not
quite well defined in its purpose. It is currently used for a number of use
cases which are typically at odds with one another.

One of the use cases of `Show` today is to provide a "debugging view" of a type.
This provides the easy ability to print *some* string representation of a type
to a stream in order to debug an application. The `Show` trait, however, is also
used for printing user-facing information. This flavor of usage is intended for
display to all users as opposed to just developers. Finally, the `Show` trait is
connected to the `ToString` trait providing the `to_string` method
unconditionally.

From these use cases of `Show`, a number of pain points have arisen over time:

1. It's not clear whether all types should implement `Show` or not. Types like
   `Path` quite intentionally avoid exposing a string representation (due to
   paths not being valid UTF-8 always) and hence do not want a `to_string`
   method to be defined on them.
2. It is quite common to use `#[deriving(Show)]` to easily print a Rust
   structure. This is not possible, however, when particular members do not
   implement `Show` (for example a `Path`).
3. Some types, such as a `String`, desire the ability to "inspect" the
   representation as well as printing the representation. An inspection mode,
   for example, would escape characters like newlines.
4. Common pieces of functionality, such as `assert_eq!` are tied to the `Show`
   trait which is not necessarily implemented for all types.

The purpose of this RFC is to clearly define what the `Show` trait is intended
to be used for, as well as providing guidelines to implementors of what
implementations should do.

# Detailed Design

As described in the motivation section, the intended use cases for the current
`Show` trait are actually motivations for two separate formatting traits. One
trait will be intended for all Rust types to implement in order to easily allow
debugging values for macros such as `assert_eq!` or general `println!`
statements. A separate trait will be intended for Rust types which are
faithfully represented as a string. These types can be represented as a string
in a non-lossy fashion and are intended for general consumption by more than
just developers.

This RFC proposes naming these two traits `Show` and `String`, respectively.

## The `String` trait

A new formatting trait will be added to `std::fmt` as follows:

```rust
pub trait String for Sized? {
    fn fmt(&self, f: &mut Formatter) -> Result;
}
```

This trait is identical to all other formatting traits except for its name. The
`String` trait will be used with the `{}` format specifier, typically considered
the default specifier for Rust.

An implementation of the `String` trait is an assertion that the type can be
faithfully represented as a UTF-8 string at all times. If the type can be
reconstructed from a string, then it is recommended, but not required, that the
following relation be true:

```rust
assert_eq!(foo, from_str(format!("{}", foo).as_slice()).unwrap());
```

If the type cannot necessarily be reconstructed from a string, then the output
may be less descriptive than the type can provide, but it is guaranteed to be
human readable for all users.

It is **not** expected that all types implement the `String` trait. Not all
types can satisfy the purpose of this trait, and for example the following types
will not implement the `String` trait:

* `Path` will abstain as it is not guaranteed to contain valid UTF-8 data.
* `CString` will abstain for the same reasons as `Path`.
* `RefCell` will abstain as it may not be accessed at all times to be
  represented as a `String`.
* `Weak` references will abstain for the same reasons as `RefCell`.

Almost all types that implement `Show` in the standard library today, however,
will implement the `String` trait. For example all primitive integer types,
vectors, slices, strings, and containers will all implement the `String` trait.
The output format will not change from what it is today (no extra escaping or
debugging will occur).

The compiler will **not** provide an implementation of `#[deriving(String)]` for
types.

## The `Show` trait

The current `Show` trait will not change location nor definition, but it will
instead move to the `{:?}` specifier instead of the `{}` specifier (which
`String` now uses).

An implementation of the `Show` trait is expected for **all** types in Rust and
provides very few guarantees about the output. Output will typically represent
the internal state as faithfully as possible, but it is not expected that this
will always be true. The output of `Show` should never be used to reconstruct
the object itself as it is not guaranteed to be possible to do so.

The purpose of the `Show` trait is to facilitate debugging Rust code which
implies that it needs to be maximally useful by extending to all Rust types. All
types in the standard library which do not currently implement `Show` will gain
an implementation of the `Show` trait including `Path`, `RefCell`, and `Weak`
references.

Many implementations of `Show` in the standard library will differ from what
they currently are today. For example `str`'s implementation will escape all
characters such as newlines and tabs in its output. Primitive integers will
print the suffix of the type after the literal in all cases. Characters will
also be printed with surrounding single quotes while escaping values such as
newlines. The purpose of these implementations are to provide debugging views
into these types.

Implementations of the `Show` trait are expected to never `panic!` and always
produce valid UTF-8 data. The compiler will continue to provide a
`#[deriving(Show)]` implementation to facilitate printing and debugging
user-defined structures.

## The `ToString` trait

Today the `ToString` trait is connected to the `Show` trait, but this RFC
proposes wiring it to the newly-proposed `String` trait instead. This switch
enables users of `to_string` to rely on the same guarantees provided by `String`
as well as not erroneously providing the `to_string` method on types that are
not intended to have one.

It is strongly discouraged to provide an implementation of the `ToString` trait
and not the `String` trait.

# Drawbacks

It is inherently easier to understand fewer concepts from the standard library
and introducing multiple traits for common formatting implementations may lead
to frequently mis-remembering which to implement. It is expected, however, that
this will become such a common idiom in Rust that it will become second nature.

This RFC establishes a convention that `Show` and `String` produce valid UTF-8
data, but no static guarantee of this requirement is provided. Statically
guaranteeing this invariant would likely involve adding some form of
`TextWriter` which we are currently not willing to stabilize for the 1.0
release.

The default format specifier, `{}`, will quickly become unable to print many
types in Rust. Without a `#[deriving]` implementation, manual implementations
are predicted to be fairly sparse. This means that the defacto default may
become `{:?}` for inspecting Rust types, providing pressure to re-shuffle the
specifiers. Currently it is seen as untenable, however, for the default output
format of a `String` to include escaped characters (as opposed to printing the
string). Due to the debugging nature of `Show`, it is seen as a non-starter to
make it the "default" via `{}`.

It may be too ambitious to define that `String` is a non-lossy representation of
a type, eventually motivating other formatting traits.

# Alternatives

The names `String` and `Show` may not necessarily imply "user readable" and
"debuggable". An alternative proposal would be to use `Show` for user
readability and `Inspect` for debugging. This alternative also opens up the door
for other names of the debugging trait like `Repr`. This RFC, however, has
chosen `String` for user readability to provide a clearer connection with the
`ToString` trait as well as emphasizing that the type can be faithfully
represented as a `String`. Additionally, this RFC considers the name `Show`
roughly on par with other alternatives and would help reduce churn for code
migrating today.

# Unresolved Questions

None at this time.
