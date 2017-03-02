- Feature Name: `try_from`
- Start Date: 2016-03-10
- RFC PR: [rust-lang/rfcs#1542](https://github.com/rust-lang/rfcs/pull/1542)
- Rust Issue: [rust-lang/rfcs#33147](https://github.com/rust-lang/rust/issues/33417)

# Summary
[summary]: #summary

The standard library provides the `From` and `Into` traits as standard ways to
convert between types. However, these traits only support *infallable*
conversions. This RFC proposes the addition of `TryFrom` and `TryInto` traits
to support these use cases in a standard way.

# Motivation
[motivation]: #motivation

Fallible conversions are fairly common, and a collection of ad-hoc traits has
arisen to support them, both [within the standard library][from-str] and [in
third party crates][into-connect-params]. A standardized set of traits
following the pattern set by `From` and `Into` will ease these APIs by
providing a standardized interface as we expand the set of fallible
conversions.

One specific avenue of expansion that has been frequently requested is fallible
integer conversion traits. Conversions between integer types may currently be
performed with the `as` operator, which will silently truncate the value if it
is out of bounds of the target type. Code which needs to down-cast values must
manually check that the cast will succeed, which is both tedious and error
prone. A fallible conversion trait reduces code like this:

```rust
let value: isize = ...;

let value: u32 = if value < 0 || value > u32::max_value() as isize {
    return Err(BogusCast);
} else {
    value as u32
};
```

to simply:

```rust
let value: isize = ...;
let value: u32 = try!(value.try_into());
```

# Detailed design
[design]: #detailed-design

Two traits will be added to the `core::convert` module:

```rust
pub trait TryFrom<T>: Sized {
    type Err;

    fn try_from(t: T) -> Result<Self, Self::Err>;
}

pub trait TryInto<T>: Sized {
    type Err;

    fn try_into(self) -> Result<T, Self::Err>;
}
```

In a fashion similar to `From` and `Into`, a blanket implementation of `TryInto`
is provided for all `TryFrom` implementations:

```rust
impl<T, U> TryInto<U> for T where U: TryFrom<T> {
    type Error = U::Err;

    fn try_into(self) -> Result<U, Self::Err> {
        U::try_from(self)
    }
}
```

In addition, implementations of `TryFrom` will be provided to convert between
*all combinations* of integer types:

```rust
#[derive(Debug)]
pub struct TryFromIntError(());

impl fmt::Display for TryFromIntError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(self.description())
    }
}

impl Error for TryFromIntError {
    fn description(&self) -> &str {
        "out of range integral type conversion attempted"
    }
}

impl TryFrom<usize> for u8 {
    type Err = TryFromIntError;

    fn try_from(t: usize) -> Result<u8, TryFromIntError> {
        // ...
    }
}

// ...
```

This notably includes implementations that are actually infallible, including
implementations between a type and itself. A common use case for these kinds
of conversions is when interacting with a C API and converting, for example,
from a `u64` to a `libc::c_long`. `c_long` may be `u32` on some platforms but
`u64` on others, so having an `impl TryFrom<u64> for u64` ensures that
conversions using these traits will compile on all architectures.  Similarly, a
conversion from `usize` to `u32` may or may not be fallible depending on the
target architecture.

The standard library provides a reflexive implementation of the `From` trait
for all types: `impl<T> From<T> for T`. We could similarly provide a "lifting"
implementation of `TryFrom`:

```rust
impl<T, U: From<T>> TryFrom<T> for U {
    type Err = Void;

    fn try_from(t: T) -> Result<U, Void> {
        Ok(U::from(t))
    }
}
```

However, this implementation would directly conflict with our goal of having
uniform `TryFrom` implementations between all combinations of integer types. In
addition, it's not clear what value such an implementation would actually
provide, so this RFC does *not* propose its addition.

# Drawbacks
[drawbacks]: #drawbacks

It is unclear if existing fallible conversion traits can backwards-compatibly
be subsumed into `TryFrom` and `TryInto`, which may result in an awkward mix of
ad-hoc traits in addition to `TryFrom` and `TryInto`.

# Alternatives
[alternatives]: #alternatives

We could avoid general traits and continue making distinct conversion traits for
each use case.

# Unresolved questions
[unresolved]: #unresolved-questions

Are `TryFrom` and `TryInto` the right names? There is some precedent for the
`try_` prefix: `TcpStream::try_clone`, `Mutex::try_lock`, etc.

What should be done about `FromStr`, `ToSocketAddrs`, and other ad-hoc fallible
conversion traits? An upgrade path may exist in the future with specialization,
but it is probably too early to say definitively.

Should `TryFrom` and `TryInto` be added to the prelude? This would be the first
prelude addition since the 1.0 release.

[from-str]: https://doc.rust-lang.org/1.7.0/std/str/trait.FromStr.html
[into-connect-params]: http://sfackler.github.io/rust-postgres/doc/v0.11.4/postgres/trait.IntoConnectParams.html
