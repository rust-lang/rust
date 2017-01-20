- Feature Name: `try_trait`
- Start Date: 2017-01-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Introduce a trait `Try` for customizing the behavior of the `?`
operator when applied to types other than `Result`.

# Motivation
[motivation]: #motivation

### Using `?` with types other than `Result`

The `?` operator is very useful for working with `Result`, but it
really applies to any sort of short-circuiting computation. As the
existence and popularity of the `try_opt!` macro confirms, it is
common to find similar patterns when working with `Option` values and
other types. Consider these two lines [from rustfmt](https://github.com/rust-lang-nursery/rustfmt/blob/29e89136957b9eedf54255c8059f8a51fbd82a68/src/expr.rs#L294-L295):

```rust
let lhs_budget = try_opt!(width.checked_sub(prefix.len() + infix.len()));
let rhs_budget = try_opt!(width.checked_sub(suffix.len()));
```

The overarching goal of this RFC is to allow lines like those to be
written using the `?` operator:

```rust
let lhs_budget = width.checked_sub(prefix.len() + infix.len())?;
let rhs_budget = width.checked_sub(suffix.len())?;
```

Naturally, this has all the advantages that `?` offered over `try!` to begin with:

- suffix notation, allowing for more fluent APIs;
- concise, yet noticeable.

However, there are some tensions to be resolved. We don't want to
hardcode the behavior of `?` to `Result` and `Option`, rather we would
like to make something more extensible. For example, futures defined
using the `futures` crate typically return one of three values:

- a successful result;
- a "not ready yet" value, indicating that the caller should try again later;
- an error.

Code working with futures typically wants to proceed only if a
successful result is returned. "Not ready yet" values as well as
errors should be propagated to the caller. This is exemplified by
[the `try_ready!` macro used in futures](https://github.com/alexcrichton/futures-rs/blob/4b027f4ac668e5024baeb51ad7146652df0b4380/src/poll.rs#L6). If
this 3-state value were written as an enum:

```rust
enum Poll<T, E> {
    Ready(T),
    NotReady,
    Error(E),
}
```

Then one could replace code like `try_ready!(self.stream.poll())` with
`self.stream.poll()?`.

(Currently, the type `Poll` in the futures crate is defined
differently, but
[alexcrichton indicates](https://github.com/rust-lang/rfcs/issues/1718#issuecomment-273323992)
that in fact the original design *did* use an `enum` like `Poll`, and
it was changed to be more compatible with the existing `try!` macro,
and hence could be changed back to be more in line with this RFC.)

### Give control over which interconversions are allowed

While it is desirable to allow `?` to be used with types other than
`Result`, **we don't want to allow arbitrary interconversion**. For
example, if `x` is a value of type `Option<T>`, we do not want to
allow `x?` to be used in a function that return a `Result<T, E>` or a
`Poll<T, E>` (from the futures example). Typically, we would only want
`x?` to be used in functions that return `Option<U>` (for some `U`).

To see why, let's consider the case where `x?` is used in a function
that returns a `Poll<T, E>`. Consider the case where `x` is `None`,
and hence we want to return early from the enclosing function. This
means we have to create a `Poll<T, E>` value to return -- but which
variant should we use?  It's not clear whether a `None` value
represents `Poll::NotReady` or `Poll::Error` (and, if the latter, it's
not clear what error!). The same applies, in a less clear fashion, to
interoperability between `Option<T>` and `Result<U, E>` -- it is not
clear whether `Some` or `None` should represent the error state, and
it's better for users to make that interconversion themselves via a
`match` or the `or_err()` method.

At the same time, there may be some cases where it makes sense to
allow interconversion between types. For example,
[a library might wish to permit a `Result<T, HttpError>` to be converted into an `HttpResponse`](https://github.com/rust-lang/rfcs/issues/1718#issuecomment-241631468)
(or vice versa). And of course the existing `?` operator allows a
`Result<T, E>` to be converted into a `Result<U, F>` so long as `F:
From<E>` (note that the types `T` and `U` are irrelevant here, as we
are only concerned with the error path). The general rule should be
that `?` can be used to interconvert between "semantically equivalent"
types. This notion of semantic equivalent is not something that can be
defined a priori in the language, and hence the design in this RFC
leaves the choice of what sorts of interconversions to enable up to
the end-user. (However, see the unresolved question at the end
concerning interactions with the orphan rules.)

# Detailed design
[design]: #detailed-design

### Desugaring and the `Try` trait

The desugaring of the `?` operator is changed to the following, where
`Try` refers to a new trait that will be introduced shortly:

```rust
match Try::try(expr) {
    Ok(v) => v,
    Err(e) => return e, // presuming no `catch` in scope
}
```

If a `catch` is in scope, the desugaring is roughly the same, except
that instead of returning, we would break out of the `catch` with `e`
as the error value.

This definition refers to a trait `Try`. This trait is defined in
`libcore` in the `ops` module; it is also mirrored in `std::ops`. The
trait `Try` is defined as follows:

```rust
trait Try<E> {
    type Success;

    /// Applies the "?" operator. A return of `Ok(t)` means that the
    /// execution should continue normally, and the result of `?` is the
    /// value `t`. A return of `Err(e)` means that execution should branch
    /// to the innermost enclosing `catch`, or return from the function.
    /// The value `e` in that case is the result to be returned.
    ///
    /// Note that the value `t` is the "unwrapped" ok value, whereas the
    /// value `e` is the *wrapped* abrupt result. So, for example, if `?`
    /// is applied to a `Result<i32, Error>`, then the types `T` and `E`
    /// here might be `T = i32` and `E = Result<(), Error>`. In
    /// particular, note that the `E` type is *not* `Error`.
    fn try(self) -> Result<Self::Success, E>;
}
```

### Initial impls

libcore will also define the following impls for the following types.

**Result**

The `Result` type includes an impl as follows:

```rust
impl<T,U,E,F> Try<Result<U,F>> for Result<T, E> where F: From<E> {
    type Success = T;

    fn try(self) -> Result<T, Result<U, F>> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(Err(F::from(e)))
        }
    }
}
```

This impl permits the `?` operator to be used on results in the same
fashion as it is used today.

**Option**

The `Option` type includes an impl as follows:

```rust
impl<T, U> Try<Option<U>> for Option<T> {
    type Success = T;

    fn try(self) -> Result<T, Option<U>> {
        match self {
            Some(v) => Ok(v),
            None => Err(None),
        }
    }
}
```

**Poll**

The `Poll` type is not included in the standard library. But just for
completeness, the equivalent of the `try_ready!` macro might be
implemented as follows:

```rust
impl<T, U, E, F> Try<Poll<U, F>> for Poll<T, E> where F: From<E> {
    type Success = T;

    fn try(self) -> Result<T, Poll<U, F>> {
        match self {
            Poll::Ready(v) => Ok(v),
            Poll::NotReady => Err(Poll::NotReady),
            Poll::Err(e) => Err(Poll::Err(F::from(e))),
        }
    }
}
```

### Interaction with type inference

Supporting more types with the `?` operator can be somewhat limiting
for type inference. In particular, if `?` only works on values of type
`Result` (as did the old `try!` macro), then `x?` forces the type of
`x` to be `Result`. This can be significant in an expression like
`vec.iter().map(|e| ...).collect()?`, since the behavior of the
`collect()` function is determined by the type it returns. In the old
`try!` macro days, `collect()` would have been forced to return a
`Result<_, _>` -- but `?` leaves it more open.

However, the impact of this should be limited, thanks to the rule
that disallows arbitrary interconversion. In particular, consider the expression
above in context:

```rust
fn foo() -> Result<X, Y> {
    let v: Vec<_> = vec.iter().map(|e| ...).collect()?;
    ...
}    
```

While it's true that `?` operator can be applied to values of many
different types, in this context it's clear that it must be applied to
a value of **some type that can be converted to a `Result<X, Y>` in
the case of failure**.  Since we don't support arbitrary
interconversion, this in fact means that the the only type which
`collect()` could have would be some sort of `Result`. **So we ought
to be able to infer the types in this example without further
annotation.** However, the current implementation fails to do so (as
can be seen in [this example](https://is.gd/v0UrMK)). This appears to
be a limitation of the trait implementation, which should be fixed
separately. (For example,
[switching the role of the type parameters in `Try` resolves the problem](https://is.gd/13A7n0).)
This merits more investigation but need not hold up the RFC.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

### Where and how to document it

This RFC proposes extending an existing operator to permit the same
general short-circuiting pattern to be used with more types. When
initially teaching the `?` operator, it would probably be best to
stick to examples around `Result`, so as to avoid confusing the
issue. However, at that time we can also mention that `?` can be
overloaded and offer a link to more comprehensive documentation, which
would show how `?` can be applied to `Option` and then explain the
desugaring and how one goes about implementing one's own impls.

The reference will have to be updated to include the new trait,
naturally.  The Rust book and Rust by example should be expanded to
include coverage of the `?` operator being used on a variety of types.

One important note is that we should develop and publish guidelines
explaining when it is appropriate to implement a `Try` interconversion
impl and when it is not (i.e., expand on the concept of semantic
equivalence used to justify why we do not permit `Option` to be
interconverted with `Result` and so forth).

### Error messages

Another important factor is the error message when `?` is used in a
function whose return type is not suitable. The current error message
in this scenario is quite opaque and directly references the `Carrer`
trait. A better message would be contingent on whether `?` is applied
to a value that implements the `Try` trait (for any return type). If
not, we can give a message like

> `?` cannot be applied to a value of type `Foo`

If, however, `?` *can* be applied to this type, but not with a function
of the given return type, we can instead report something like this:

> cannot use the `?` operator in a function that returns `()`

or perhaps if we want to be more strictly correct:

> `?` cannot be applied to a `io::Result` in a function that returns `()`

We could also go further and analyze the type to which `?` is applied
and figure out the set of legal return types for the function to
have. So if the code is invoking `foo.write()?` (i.e., applying `?` to an
`io::Result`), then we could offer a suggestion like "consider changing
the return type to `Result<(), io::Error>`" or perhaps just "consider
changing the return type to a `Result"`. 

# Drawbacks
[drawbacks]: #drawbacks

One drawback of supporting more types is that type inference becomes
harder. This is because an expression like `x?` no longer implies that
the type of `x` is `Result`. However, type inference is still expected
to work well in practice, as discussed in the detailed design section.

# Alternatives
[alternatives]: #alternatives

### Using an associated type for the success value

The proposed `Try` trait has one generic type parameter (`E`) which
encodes the type to return on error. The type to return on success is
encoded as an associated type (`type Success`). This implies that the
type of the expression `x?` (i.e., the `Success` type) is going to be
determined by the type of `x` (as well as the return type `E`). An
alternative formulation of the `Try` trait used two generic type
parameters, one for success and one for error:

```rust
trait Try<S, E> {
    fn try(self) -> Result<S, E>
}
```

In this formulation, the type of `x?` could be influenced by both the
return type `E` as well as the type to which `x?` is being coerced.
This implies that e.g. one could make a version of `?` that uses
`Into` implicitly both on the success *and* the error values:

```Rust
impl<T,U,V,E,F> Try<V, Result<U,F>> for Result<T, E> where F: From<E>, V: From<T> {
    fn try(self) -> Result<T, Result<U, F>> {
        match self {
            Ok(t) => Ok(V::from(t)),
            Err(e) => Err(Err(F::from(e)))
        }
    }
}
```

In general, having this flexibility seemed undesirable, since it would
be surprising for `x?` to perform coercions on the unwrapped value on
the success path, and it would also potentially present an inference
challenge, since the type of `x?` would not necessarily be uniquely
determined by the type of `x`.

In fact, if we wanted to ensure that the type of `x?` is determined
*solely* by `x` and not by the surrouding return type `E`, we might
also consider introducing two traits:

```rust
trait Try<E>: TrySuccess {
    fn try(self) -> Result<Self::Success, E>;
}

trait TrySuccess {
    type Success;
}
```

One would then implement these traits as follows:

```rust
impl<T, U> Try<Option<U>> for Option<T> { ... }

impl<T> TrySuccess for Option<T> {
    type Success = T;
}
```

Note in particular the second impl, which shows that the type
`Success` is defined purely in terms of the `Option<T>` to which the
`?` is applied, and not the `Option<U>` that it returns in the case of
error.

### Using a distinct return type

The `Try` trait uses `Result` as its return type. It has also been proposed
that we could introduce a distinct enum type for the return value. For example:

```rust
enum TryResult<T, E> {
    Ok(T),
    Abrupt(E),
}
```

Re-using `Result` was chosen because it is simpler (fewer things being
added).  It also allows manual invocations of `Try` (should there by
any, perhaps in macros) to re-use the rich set of methods available on
`Result`. Finally, the `Result` type seems semantically
appropriate. In general, `Result` is used to indicate whether normal
execution can continue (`Ok`) or whether some form of recoverable
error occurred (`Err`).  In this case, "normal execution" means the
rest of the function, and the recoverable error means abrupt
termination:

- returning a `Ok(v)` value means that (a) execution
  should continue normally and (b) the result `v` represents the value to
  be propagated;
- returning a `Err(e)` value means that (a) execution should return
  abruptly and (b) the result `e` is the value to be propagated (note
  that `e` may itself be a `Result`, if the function returns
  `Result`).
  
### The original `Carrier` trait proposal

The original `Carrier` as proposed in RFC 243 had a rather different
design:

```rust
trait ResultCarrier {
    type Normal;
    type Exception;
    fn embed_normal(from: Normal) -> Self;
    fn embed_exception(from: Exception) -> Self;
    fn translate<Other: ResultCarrier<Normal=Normal, Exception=Exception>>(from: Self) -> Other;
}
```

Whereas this `Try` trait links the type of the value being matched
(`Self`) with the type that the enclosing function returns (`E`), the
`Carrier` was implemented separately for each type (e.g., `Result` and
`Option`). It allowed any kind of carrier to be converted to any other
kind of carrier, which fails to preserve the "semantic equivalent"
property.  It would also interact poorly with type inference, as
discussed in the "Detailed Design" section.

### Traits implemented over higher-kinded types

The "semantic equivalent" property might suggest that the `Carrier`
trait ought to be defined over higher-kinded types (or generic
associated types) in some form. The most obvious downside of such a
design is that Rust does not offer higher-kinded types nor anything
equivalent to them today, and hence we would have to block on that
design effort. But it also turns out that HKT is
[not a particularly good fit for the problem](https://github.com/rust-lang/rust/pull/35056#issuecomment-240129923). To
start, consider what "kind" the `Self` parameter on the `Try` trait
would have to have.  If we were to implement `Try` on `Option`, it
would presumably then have kind `type -> type`, but we also wish to
implement `Try` on `Result`, which has kind `type -> type ->
type`. There has even been talk of implementing `Try` for simple types
like `bool`, which simply have kind `type`. More generally, the
problems encountered are quite similar to the problems that
[Simon Peyton-Jones describes in attempting to model collections using HKT](https://github.com/rust-lang/rust/pull/35056#issuecomment-240129923):
we wish the `Try` trait to be implemented in a great number of
scenarios.  Some of them, like converting `Result<T,E>` to
`Result<U,F>`, allow for the type of the success value and the error
value to both be changed, though not arbitrarily (subject to the
`From` trait, in particular).  Others, like converting `Option<T>` to
`Option<U>`, allow only the type of the success value to change,
whereas others (like converting `bool` to `bool`) do not allow either
type to change.

### What to name the trait

A number of names have been proposed for this trait. The original name
was `Carrier`, as the implementing type was the "carrier" for an error
value. A proposed alternative was `QuestionMark`, named after the
operator `?`. However, the general consensus seemed to be that since
Rust operator overloading traits tend to be named after the
*operation* that the operator performed (e.g., `Add` and not `Plus`,
`Deref` and not `Star` or `Asterix`), it was more appropriate to name
the trait `Try`, which seems to be the best name for the operation in
question.

# Unresolved questions
[unresolved]: #unresolved-questions

**We need to resolve the interactions with type inference.** It is
important that expressions like `vec.iter().map(|e| ...).collect()?`
are able to infer the type of `collect()` from context. This may
require small tweaks to the `Try` trait, though the author considers
that unlikely.

**Should we reverse the order of the trait's type parameters?** The
current ordering of the trait type parameters seems natural, since
`Self` refers to the type of the value to which `?` is
applied. However, it has
[negative interactions with the orphan rules](https://github.com/rust-lang/rfcs/issues/1718#issuecomment-273353457).
In particular, it means that one cannot write an impl that converts a
`Result` into any other type. For example, in the futures library, it
would be nice to have an impl that allows a result to be converted
into a `Poll`:

```rust
impl<T, U, E, F> Try<Poll<U, F>> for Result<T, E>
    where F: From<E>
{ }
```

However, this would fall afoul of the current orphan rules. If we
switched the order of the trait's type parameters, then this impl
would be allowed, but of course other impls would not be (e.g.,
something which allowed a `Poll` to be converted into a `Result`,
although that particular conversion would not make sense). Certainly
this is a more general problem with the orphan rules that we might
want to consider resolving in a more general way, and indeed
[specialization may provide a solution]. But it may still be worth
considering redefining the trait to sidestep the problem in the short
term. If we chose to do so, the trait would look like:

```rust
trait Try<V> {
    type Success;
    fn try(value: V) -> Result<Self::Success, Self>;
}
```

[specialization may provide a solution]: https://github.com/rust-lang/rfcs/issues/1718#issuecomment-273415458
