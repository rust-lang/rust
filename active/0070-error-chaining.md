- Start Date: (fill me in with today's date, 2014-07-17)
- RFC PR #: [rust-lang/rfcs#201](https://github.com/rust-lang/rfcs/pull/201)
- Rust Issue #: [rust-lang/rust#17747](https://github.com/rust-lang/rust/issues/17747)

# Summary

This RFC improves interoperation between APIs with different error
types. It proposes to:

* Increase the flexibility of the `try!` macro for clients of multiple
  libraries with disparate error types.

* Standardize on basic functionality that any error type should have
  by introducing an `Error` trait.

* Support easy error chaining when crossing abstraction boundaries.

The proposed changes are all library changes; no language changes are
needed -- except that this proposal depends on
[multidispatch](https://github.com/rust-lang/rfcs/pull/195) happening.

# Motivation

Typically, a module (or crate) will define a custom error type encompassing the
possible error outcomes for the operations it provides, along with a custom
`Result` instance baking in this type. For example, we have `io::IoError` and
`io::IoResult<T> = Result<T, io::IoError>`, and similarly for other libraries.
Together with the `try!` macro, the story for interacting with errors for a
single library is reasonably good.

However, we lack infrastructure when consuming or building on errors from
multiple APIs, or abstracting over errors.

## Consuming multiple error types

Our current infrastructure for error handling does not cope well with
mixed notions of error.

Abstractly, as described by
[this issue](https://github.com/rust-lang/rust/issues/14419), we
cannot do the following:

```
fn func() -> Result<T, Error> {
    try!(may_return_error_type_A());
    try!(may_return_error_type_B());
}
```

Concretely, imagine a CLI application that interacts both with files
and HTTP servers, using `std::io` and an imaginary `http` crate:

```
fn download() -> Result<(), CLIError> {
    let contents = try!(http::get(some_url));
    let file = try!(File::create(some_path));
    try!(file.write_str(contents));
    Ok(())
}
```

The `download` function can encounter both `io` and `http` errors, and
wants to report them both under the common notion of `CLIError`. But
the `try!` macro only works for a single error type at a time.

There are roughly two scenarios where multiple library error types
need to be coalesced into a common type, each with different needs:
application error reporting, and library error reporting

### Application error reporting: presenting errors to a user

An application is generally the "last stop" for error handling: it's
the point at which remaining errors are presented to the user in some
form, when they cannot be handled programmatically.

As such, the data needed for application-level errors is usually
related to human interaction. For a CLI application, a short text
description and longer verbose description are usually all that's
needed. For GUI applications, richer data is sometimes required, but
usually not a full `enum` describing the full range of errors.

Concretely, then, for something like the `download` function above,
for a CLI application, one might want `CLIError` to roughly be:

```rust
struct CLIError<'a> {
    description: &'a str,
    detail: Option<String>,
    ... // possibly more fields here; see detailed design
}
```

Ideally, one could use the `try!` macro as in the `download` example
to coalesce a variety of error types into this single, simple
`struct`.

### Library error reporting: abstraction boundaries

When one library builds on others, it needs to translate from their
error types to its own. For example, a web server framework may build
on a library for accessing a SQL database, and needs some way to
"lift" SQL errors to its own notion of error.

In general, a library may not want to reveal the upstream libraries it
relies on -- these are implementation details which may change over
time. Thus, it is critical that the error type of upstream libraries
not leak, and "lifting" an error from one library to another is a way
of imposing an abstraction boundaries.

In some cases, the right way to lift a given error will depend on the
operation and context. In other cases, though, there will be a general
way to embed one kind of error in another (usually via a
["cause chain"](http://docs.oracle.com/javase/tutorial/essential/exceptions/chained.html)). Both
scenarios should be supported by Rust's error handling infrastructure.

## Abstracting over errors

Finally, libraries sometimes need to work with errors in a generic
way. For example, the `serialize::Encoder` type takes is generic over
an arbitrary error type `E`. At the moment, such types are completely
arbitrary: there is no `Error` trait giving common functionality
expected of all errors. Consequently, error-generic code cannot
meaningfully interact with errors.

(See [this issue](https://github.com/rust-lang/rust/issues/15036) for
a concrete case where a bound would be useful; note, however, that the
design below does not cover this use-case, as explained in
Alternatives.)

Languages that provide exceptions often have standard exception
classes or interfaces that guarantee some basic functionality,
including short and detailed descriptions and "causes". We should
begin developing similar functionality in `libstd` to ensure that we
have an agreed-upon baseline error API.

# Detailed design

We can address all of the problems laid out in the Motivation section
by adding some simple library code to `libstd`, so this RFC will
actually give a full implementation.

**Note**, however, that this implementation relies on the
[multidispatch](https://github.com/rust-lang/rfcs/pull/195) proposal
currently under consideration.

The proposal consists of two pieces: a standardized `Error` trait and
extensions to the `try!` macro.

## The `Error` trait

The standard `Error` trait follows very the widespread pattern found
in `Exception` base classes in many languages:

```rust
pub trait Error: Send + Any {
    fn description(&self) -> &str;

    fn detail(&self) -> Option<&str> { None }
    fn cause(&self) -> Option<&Error> { None }
}
```

Every concrete error type should provide at least a description. By
making this a slice-returning method, it is possible to define
lightweight `enum` error types and then implement this method as
returning static string slices depending on the variant.

The `cause` method allows for cause-chaining when an error crosses
abstraction boundaries. The cause is recorded as a trait object
implementing `Error`, which makes it possible to read off a kind of
abstract backtrace (often more immediately helpful than a full
backtrace).

The `Any` bound is needed to allow *downcasting* of errors. This RFC
stipulates that it must be possible to downcast errors in the style of
the `Any` trait, but leaves unspecified the exact implementation
strategy.  (If trait object upcasting was available, one could simply
upcast to `Any`; otherwise, we will likely need to duplicate the
`downcast` APIs as blanket `impl`s on `Error` objects.)

It's worth comparing the `Error` trait to the most widespread error
type in `libstd`, `IoError`:

```rust
pub struct IoError {
    pub kind: IoErrorKind,
    pub desc: &'static str,
    pub detail: Option<String>,
}
```

Code that returns or asks for an `IoError` explicitly will be able to
access the `kind` field and thus react differently to different kinds
of errors. But code that works with a generic `Error` (e.g.,
application code) sees only the human-consumable parts of the error.
In particular, application code will often employ `Box<Error>` as the
error type when reporting errors to the user. The `try!` macro
support, explained below, makes doing so ergonomic.

## An extended `try!` macro

The other piece to the proposal is a way for `try!` to automatically
convert between different types of errors.

The idea is to introduce a trait `FromError<E>` that says how to
convert from some lower-level error type `E` to `Self`. The `try!`
macro then passes the error it is given through this conversion before
returning:

```rust
// E here is an "input" for dispatch, so conversions from multiple error
// types can be provided
pub trait FromError<E> {
    fn from_err(err: E) -> Self;
}

impl<E> FromError<E> for E {
    fn from_err(err: E) -> E {
        err
    }
}

impl<E: Error> FromError<E> for Box<Error> {
    fn from_err(err: E) -> Box<Error> {
        box err as Box<Error>
    }
}

macro_rules! try (
    ($expr:expr) => ({
        use error;
        match $expr {
            Ok(val) => val,
            Err(err) => return Err(error::FromError::from_err(err))
        }
    })
)
```

This code depends on
[multidispatch](https://github.com/rust-lang/rfcs/pull/195), because
the conversion depends on both the source and target error types.  (In
today's Rust, the two implementations of `FromError` given above would
be considered overlapping.)

Given the blanket `impl` of `FromError<E>` for `E`, all existing uses
of `try!` would continue to work as-is.

With this infrastructure in place, application code can generally use
`Box<Error>` as its error type, and `try!` will take care of the rest:

```
fn download() -> Result<(), Box<Error>> {
    let contents = try!(http::get(some_url));
    let file = try!(File::create(some_path));
    try!(file.write_str(contents));
    Ok(())
}
```

Library code that defines its own error type can define custom
`FromError` implementations for lifting lower-level errors (where the
lifting should also perform cause chaining) -- at least when the
lifting is uniform across the library. The effect is that the mapping
from one error type into another only has to be written one, rather
than at every use of `try!`:

```
impl FromError<ErrorA> MyError { ... }
impl FromError<ErrorB> MyError { ... }

fn my_lib_func() -> Result<T, MyError> {
    try!(may_return_error_type_A());
    try!(may_return_error_type_B());
}
```

# Drawbacks

The main drawback is that the `try!` macro is a bit more complicated.

# Unresolved questions

## Conventions

This RFC does not define any particular conventions around cause
chaining or concrete error types. It will likely take some time and
experience using the proposed infrastructure before we can settle
these conventions.

## Extensions

The functionality in the `Error` trait is quite minimal, and should
probably grow over time. Some additional functionality might include:

### Features on the `Error` trait

* **Generic creation of `Error`s.** It might be useful for the `Error`
  trait to expose an associated constructor. See
  [this issue](https://github.com/rust-lang/rust/issues/15036) for an
  example where this functionality would be useful.

* **Mutation of `Error`s**. The `Error` trait could be expanded to
  provide setters as well as getters.

The main reason not to include the above two features is so that
`Error` can be used with extremely minimal data structures,
e.g. simple `enum`s. For such data structures, it's possible to
produce fixed descriptions, but not mutate descriptions or other error
properties. Allowing generic creation of any `Error`-bounded type
would also require these `enum`s to include something like a
`GenericError` variant, which is unfortunate. So for now, the design
sticks to the least common denominator.

### Concrete error types

On the other hand, for code that doesn't care about the footprint of
its error types, it may be useful to provide something like the
following generic error type:

```rust
pub struct WrappedError<E> {
    pub kind: E,
    pub description: String,
    pub detail: Option<String>,
    pub cause: Option<Box<Error>>
}

impl<E: Show> WrappedError<E> {
    pub fn new(err: E) {
        WrappedErr {
            kind: err,
            description: err.to_string(),
            detail: None,
            cause: None
        }
    }
}

impl<E> Error for WrappedError<E> {
    fn description(&self) -> &str {
        self.description.as_slice()
    }
    fn detail(&self) -> Option<&str> {
        self.detail.as_ref().map(|s| s.as_slice())
    }
    fn cause(&self) -> Option<&Error> {
        self.cause.as_ref().map(|c| &**c)
    }
}
```

This type can easily be added later, so again this RFC sticks to the
minimal functionality for now.
