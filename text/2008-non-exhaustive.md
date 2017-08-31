- Feature Name: non_exhaustive
- Start Date: 2017-05-24
- RFC PR: https://github.com/rust-lang/rfcs/pull/2008
- Rust Issue: https://github.com/rust-lang/rust/issues/44109

# Summary

This RFC introduces the `#[non_exhaustive]` attribute for enums and structs,
which indicates that more variants/fields may be added to an enum/struct in the
future.

Adding this hint to enums will force downstream crates to add a wildcard arm to
`match` statements, ensuring that adding new variants is not a breaking change.

Adding this hint to structs or enum variants will prevent downstream crates
from constructing or exhaustively matching, to ensure that adding new fields is
not a breaking change.

This is a post-1.0 version of [RFC 757], with some additions.

# Motivation

## Enums

The most common use for non-exhaustive enums is error types. Because adding
features to a crate may result in different possibilities for errors, it makes
sense that more types of errors will be added in the future.

For example, the rustdoc for [`std::io::ErrorKind`] shows:

```rust
pub enum ErrorKind {
    NotFound,
    PermissionDenied,
    ConnectionRefused,
    ConnectionReset,
    ConnectionAborted,
    NotConnected,
    AddrInUse,
    AddrNotAvailable,
    BrokenPipe,
    AlreadyExists,
    WouldBlock,
    InvalidInput,
    InvalidData,
    TimedOut,
    WriteZero,
    Interrupted,
    Other,
    UnexpectedEof,
    // some variants omitted
}
```

Because the standard library continues to grow, it makes sense to eventually add
more error types. However, this can be a breaking change if we're not careful;
let's say that a user does a match statement like this:

```rust
use std::io::ErrorKind::*;

match error_kind {
    NotFound => ...,
    PermissionDenied => ...,
    ConnectionRefused => ...,
    ConnectionReset => ...,
    ConnectionAborted => ...,
    NotConnected => ...,
    AddrInUse => ...,
    AddrNotAvailable => ...,
    BrokenPipe => ...,
    AlreadyExists => ...,
    WouldBlock => ...,
    InvalidInput => ...,
    InvalidData => ...,
    TimedOut => ...,
    WriteZero => ...,
    Interrupted => ...,
    Other => ...,
    UnexpectedEof => ...,
}
```

If we were to add another variant to this enum, this `match` would fail,
requiring an additional arm to handle the extra case. But, if force users to
add an arm like so:

```rust
match error_kind {
    // ...
    _ => ...,
}
```

Then we can add as many variants as we want without breaking any downstream
matches.

### How we do this today

We force users add this arm for [`std::io::ErrorKind`] by adding a hidden
variant:

```rust
#[unstable(feature = "io_error_internals",
           reason = "better expressed through extensible enums that this \
                     enum cannot be exhaustively matched against",
           issue = "0")]
#[doc(hidden)]
__Nonexhaustive,
```

Because this feature doesn't show up in the docs, and doesn't work in stable
rust, we can safely assume that users won't use it.

A lot of crates take advantage of `#[doc(hidden)]` variants to tell users that
they should add a wildcard branch to matches. However, the standard library
takes this trick further by making the variant `unstable`, ensuring that it
cannot be used in stable Rust. Outside the standard library, here's a look at
[`diesel::result::Error`]:

```rust
pub enum Error {
    InvalidCString(NulError),
    DatabaseError(String),
    NotFound,
    QueryBuilderError(Box<StdError+Send+Sync>),
    DeserializationError(Box<StdError+Send+Sync>),
    #[doc(hidden)]
    __Nonexhaustive,
}
```

Even though the variant is hidden in the rustdoc, there's nothing actually
stopping a user from using the `__Nonexhaustive` variant. This code works
totally fine, for example:

```rust
use diesel::Error::*;
match error {
    InvalidCString(..) => ...,
    DatabaseError(..) => ...,
    NotFound => ...,
    QueryBuilderError(..) => ...,
    DeserializationError(..) => ...,
    __Nonexhaustive => ...,
}
```

This seems unintended, even though this is currently the best way to make
non-exhaustive enums outside the standard library. In fact, even the standard
library remarks that this is a hack. Recall the hidden variant for
[`std::io::ErrorKind`]:

```rust
#[unstable(feature = "io_error_internals",
           reason = "better expressed through extensible enums that this \
                     enum cannot be exhaustively matched against",
           issue = "0")]
#[doc(hidden)]
__Nonexhaustive,
```

Using `#[doc(hidden)]` will forever feel like a hack to fix this problem.
Additionally, while plenty of crates could benefit from the idea of
non-exhaustiveness, plenty don't because this isn't documented in the Rust book,
and only documented elsewhere as a hack until a better solution is proposed.

### Opportunity for optimisation

Currently, the `#[doc(hidden)]` hack leads to a few missed opportunities
for optimisation. For example, take this enum:

```rust
pub enum Error {
    Message(String),
    Other,
}
```

Currently, this enum takes up the same amount of space as `String` because of
the non-zero optimisation. If we add our non-exhaustive variant:

```rust
pub enum Error {
    Message(String),
    Other,
    #[doc(hidden)]
    __Nonexhaustive,
}
```

Then this enum needs an extra bit to distinguish `Other` and `__Nonexhaustive`,
which is ultimately never used. This will likely add an extra 8 bytes on a
64-bit system to ensure alignment.

More importantly, take the following code:

```rust
use Error::*;
match error {
    Message(ref s) => /* lots of code */,
    Other => /* lots of code */,
    _ => /* lots of code */,
}
```

As a human, we can determine that the wildcard match is dead code and can be
removed from the binary. Unfortunately, Rust can't make this distinction because
we could still *technically* use that wildcard branch.

Although these options will unlikely matter in this example because
error-handling code (hopefully) shouldn't run very often, it could matter for
other use cases.

## Structs

The most common use for non-exhaustive structs is config types. It often makes
sense to make fields public for ease-of-use, although this can ultimately lead
to breaking changes if we're not careful.

For example, take this config struct:

```rust
pub struct Config {
    pub window_width: u16,
    pub window_height: u16,
}
```

As this configuration struct gets larger, it makes sense that more fields will
be added. In the future, the crate may decide to add more public fields, or some
private fields. For example, let's assume we make the following addition:

```rust
pub struct Config {
    pub window_width: u16,
    pub window_height: u16,
    pub is_fullscreen: bool,
}
```

Now, code that constructs the struct, like below, will fail to compile:

```
let config = Config { window_width: 640, window_height: 480 };
```

And code that matches the struct, like below, will also fail to compile:

```rust
if let Ok(Config { window_width, window_height }) = load_config() {
    // ...
}
```

Adding this new setting is now a breaking change! To rectify this, we could
always add a private field:

```rust
pub struct Config {
    pub window_width: u16,
    pub window_height: u16,
    pub is_fullscreen: bool,
    non_exhaustive: (),
}
```

But this makes it more difficult for the crate itself to construct `Config`,
because you have to add a `non_exhaustive: ()` field every time you make a new
value.

### Other kinds of structs

Because enum variants are *kind* of like a struct, any change we make to structs
should apply to them too. Additionally, any change should apply to tuple structs
as well.

# Detailed design

An attribute `#[non_exhaustive]` is added to the language, which will (for now)
fail to compile if it's used on anything other than an enum or struct
definition, or enum variant.

## Enums

Within the crate that defines the enum, this attribute is essentially ignored,
so that the current crate can continue to exhaustively match the enum. The
justification for this is that any changes to the enum will likely result in
more changes to the rest of the crate. Consider this example:

```rust
use std::error::Error as StdError;

#[non_exhaustive]
pub enum Error {
    Message(String),
    Other,
}
impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Message(ref s) => s,
            Other => "other or unknown error",
        }
    }
}
```

It seems undesirable for the crate author to use a wildcard arm here, to
ensure that an appropriate description is given for every variant. In fact, if
they use a wildcard arm in addition to the existing variants, it should be
identified as dead code, because it will never be run.

Outside the crate that defines the enum, users should be required to add a
wildcard arm to ensure forward-compatibility, like so:

```rust
use mycrate::Error;

match error {
    Message(ref s) => ...,
    Other => ...,
    _ => ...,
}
```

And it should *not* be marked as dead code, even if the compiler does mark it as
dead and remove it.

Note that this can *potentially* cause breaking changes if a user adds
`#[deny(dead_code)]` to a match statement *and* the upstream crate removes the
`#[non_exhaustive]` lint. That said, modifying warn-only lints is generally
assumed to not be a breaking change, even though users can make it a breaking
change by manually denying lints.

## Structs

Like with enums, the attribute is essentially ignored in the crate that defines
the struct, so that users can continue to construct values for the struct.
However, this will prevent downstream users from constructing or exhaustively
matching the struct, because fields may be added to the struct in the future.

Additionally, adding `#[non_exhaustive]` to an enum variant will operate exactly
the same as if the variant were a struct.

Using our `Config` again:

```rust
#[non_exhaustive]
pub struct Config {
    pub window_width: u16,
    pub window_height: u16,
}
```

We can still construct our config within the defining crate like so:

```rust
let config = Config { window_width: 640, window_height: 480 };
```

And we can even exhaustively match on it, like so:

```rust
if let Ok(Config { window_width, window_height }) = load_config() {
    // ...
}
```

But users outside the crate won't be able to construct their own values, because
otherwise, adding extra fields would be a breaking change.

Users can still match on `Config`s non-exhaustively, as usual:

```rust
let &Config { window_width, window_height, .. } = config;
```

But without the `..`, this code will fail to compile.

Although it should not be explicitly forbidden by the language to mark a struct
with some private fields as non-exhaustive, it should emit a warning to tell the
user that the attribute has no effect.

## Tuple structs

Non-exhaustive tuple structs will operate similarly to structs, however, will
disallow matching directly. For example, take this example on stable today:

```rust
pub Config(pub u16, pub u16, ());
```

The below code does not work, because you can't match tuple structs with private
fields:

```rust
let Config(width, height, ..) = config;
```

However, this code *does* work:

```rust
let Config { 0: width, 1: height, .. } = config;
```

So, if we label a struct non-exhaustive:

```
#[non_exhaustive]
pub Config(pub u16, pub u16)
```

Then we the only valid way of matching will be:

```rust
let Config { 0: width, 1: height, .. } = config;
```

We can think of this as lowering the visibility of the constructor to
`pub(crate)` if it is marked as `pub`, then applying the standard structure
rules.

## Unit structs

Unit structs will work very similarly to tuple structs. Consider this struct:

```rust
#[non_exhaustive]
pub struct Unit;
```

We won't be able to construct any values of this struct, but we will be able to
match it like:

```rust
let Unit { .. } = unit;
```

Similarly to tuple structs, this will simply lower the visibility of the
constructor to `pub(crate)` if it were marked as `pub`.

## Functional record updates

Functional record updates will operate very similarly to if the struct had an
extra, private field. Take this example:

```
#[derive(Debug)]
#[non_exhaustive]
pub struct Config {
    pub width: u16,
    pub height: u16,
    pub fullscreen: bool,
}
impl Default for Config {
    fn default() -> Config {
        Config { width: 640, height: 480, fullscreen: false }
    }
}
```

We'd expect this code to work without the `non_exhaustive` attribute:

```
let c = Config { width: 1920, height: 1080, ..Config::default() };
println!("{:?}", c);
```

Although outside of the defining crate, it will not, because `Config` could, in
the future, contain private fields that the user didn't account for.

## Changes to rustdoc

Right now, the only indicator that rustdoc gives for non-exhaustive enums and
structs is a comment saying "some variants/fields omitted." This shows up
whenever variants or fields are marked as `#[doc(hidden)]`, or when fields are
private. rustdoc should continue to emit this message in these cases.

However, after this message (if any), it should offer an additional message
saying "more variants/fields may be added in the future," to clarify that the
enum/struct is non-exhaustive. It also hints to the user that in the future,
they may want to fine-tune any match code for enums to include future variants
when they are added.

These two messages should be distinct; the former says "this enum/struct has
stuff that you shouldn't see," while the latter says "this enum/struct is
incomplete and may be extended in the future."

# How We Teach This

Changes to rustdoc should make it easier for users to understand the concept of
non-exhaustive enums and structs in the wild.

In the chapter on enums, a section should be added specifically for
non-exhaustive enums. Because error types are common in almost all crates, this
case is important enough to be taught when a user learns Rust for the first
time.

Additionally, non-exhaustive structs should be documented in an early chapter on
structs. Public fields should be preferred over getter/setter methods in Rust,
although users should be aware that adding extra fields is a potentially
breaking change. In this chapter, users should be taught about non-exhaustive
enum variants as well.

# Drawbacks

* The `#[doc(hidden)]` hack in practice is usually good enough.
* An attribute may be more confusing than a dedicated syntax.
* `non_exhaustive` may not be the clearest name.

# Alternatives

* Provide a dedicated syntax instead of an attribute. This would likely be done
  by adding a `...` variant or field, as proposed by the original
  [extensible enums RFC][RFC 757].
* Allow creating private enum variants and/or private fields for enum variants,
  giving a less-hacky way to create a hidden variant/field.
* Document the `#[doc(hidden)]` hack and make it more well-known.

# Unresolved questions

It may make sense to have a "not exhaustive enough" lint to non-exhaustive
enums or structs, so that users can be warned if they are missing fields or
variants despite having a wildcard arm to warn on them.

Although this is beyond the scope of this particular RFC, it may be good as a
clippy lint in the future.

## Extending to traits

Tangentially, it also makes sense to have non-exhaustive traits as well, even
though they'd be non-exhaustive in a different way. Take this example from
[`byteorder`]:

```rust
pub trait ByteOrder: Clone + Copy + Debug + Default + Eq + Hash + Ord + PartialEq + PartialOrd {
   // ...
}
```

The `ByteOrder` trait requires these traits so that a user can simply write a
bound of `T: ByteOrder` without having to add other useful traits, like `Hash`
or `Eq`.

This trait is useful, but the crate has no intention of letting other users
implement this trait themselves, because then adding an additional trait
dependency for `ByteOrder` could be a breaking change.

The way that this crate solves this problem is by adding a hidden trait
dependency:

```rust
mod private {
    pub trait Sealed {}
    impl Sealed for super::LittleEndian {}
    impl Sealed for super::BigEndian {}
}

pub trait ByteOrder: /* ... */ + private::Sealed {
    // ...
}
```

This way, although downstream crates can use this trait, they cannot actually
implement things for this trait.

This pattern could again be solved by using `#[non_exhaustive]`:

```rust
#[non_exhaustive]
pub trait ByteOrder: /* ... */ {
    // ...
}
```

This would indicate to downstream traits that this trait might gain additional
requirements (dependent traits or methods to implement), and as such, cannot be
implemented downstream.

[RFC 757]: https://github.com/rust-lang/rfcs/pull/757
[`std::io::ErrorKind`]: https://doc.rust-lang.org/1.17.0/std/io/enum.ErrorKind.html
[`diesel::result::Error`]: https://docs.rs/diesel/0.13.0/diesel/result/enum.Error.html
[use clauses]: https://github.com/rust-lang/rfcs/pull/1976#issuecomment-301903528
[`byteorder`]: https://github.com/BurntSushi/byteorder/tree/f8e7685b3a81c52f5448fd77fb4e0535bc92f880
