- Feature Name: `try_trait`
- Start Date: 2017-01-19
- RFC PR: [rust-lang/rfcs#1859](https://github.com/rust-lang/rfcs/pull/1859)
- Rust Issue: [rust-lang/rust#31436](https://github.com/rust-lang/rust/issues/31436)

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

### Support interconversion, but with caution

The existing `try!` macro and `?` operator already allow a limit
amount of type conversion, specifically in the error case. That is, if
you apply `?` to a value of type `Result<T, E>`, the surrouding
function can have some other return type `Result<U, F>`, so long as
the error types are related by the `From` trait (`F: From<E>`). The
idea is that if an error occurs, we will wind up returning
`F::from(err)`, where `err` is the actual error. This is used (for
example) to "upcast" various errors that can occur in a function into
a common error type (e.g., `Box<Error>`).

In some cases, it would be useful to be able to convert even more
freely. At the same time, there may be some cases where it makes sense
to allow interconversion between types. For example,
[a library might wish to permit a `Result<T, HttpError>` to be converted into an `HttpResponse`](https://github.com/rust-lang/rfcs/issues/1718#issuecomment-241631468)
(or vice versa). Or, in the futures example given above, we might wish
to apply `?` to a `Poll` value and use that in a function that itself
returns a `Poll`:

```rust
fn foo() -> Poll<T, E> {
    let x = bar()?; // propagate error case
}
```

and we might wish to do the same, but in a function returning a `Result`:

```rust
fn foo() -> Result<T, E> {
    let x = bar()?; // propagate error case
}
```

However, we wish to be sure that this sort of interconversion is
*intentional*. In particular, `Result` is often used with a semantic
intent to mean an "unhandled error", and thus if `?` is used to
convert an error case into a "non-error" type (e.g., `Option`), there
is a risk that users accidentally overlook error cases. To mitigate
this risk, we adopt certain conventions (see below) in that case to
help ensure that "accidental" interconversion does not occur.

# Detailed design
[design]: #detailed-design

### Playground

Note: if you wish to experiment,
[this Rust playgroud link](https://play.rust-lang.org/?gist=9ef8effa0c1c81bc8bb8dccb07505c54&version=stable&backtrace=0)
contains the traits and impls defined herein.

### Desugaring and the `Try` trait

The desugaring of the `?` operator is changed to the following, where
`Try` refers to a new trait that will be introduced shortly:

```rust
match Try::into_result(expr) {
    Ok(v) => v,

    // here, the `return` presumes that there is
    // no `catch` in scope:
    Err(e) => return Try::from_error(From::from(e)),
}
```

If a `catch` is in scope, the desugaring is roughly the same, except
that instead of returning, we would break out of the `catch` with `e`
as the error value.

This definition refers to a trait `Try`. This trait is defined in
`libcore` in the `ops` module; it is also mirrored in `std::ops`. The
trait `Try` is defined as follows:

```rust
trait Try {
    type Ok;
    type Error;
    
    /// Applies the "?" operator. A return of `Ok(t)` means that the
    /// execution should continue normally, and the result of `?` is the
    /// value `t`. A return of `Err(e)` means that execution should branch
    /// to the innermost enclosing `catch`, or return from the function.
    ///
    /// If an `Err(e)` result is returned, the value `e` will be "wrapped"
    /// in the return type of the enclosing scope (which must itself implement
    /// `Try`). Specifically, the value `X::from_error(From::from(e))`
    /// is returned, where `X` is the return type of the enclosing function.
    fn into_result(self) -> Result<Self::Ok, Self::Error>;

    /// Wrap an error value to construct the composite result. For example,
    /// `Result::Err(x)` and `Result::from_error(x)` are equivalent.
    fn from_error(v: Self::Error) -> Self;

    /// Wrap an OK value to construct the composite result. For example,
    /// `Result::Ok(x)` and `Result::from_ok(x)` are equivalent.
    ///
    /// *The following function has an anticipated use, but is not used
    /// in this RFC. It is included because we would not want to stabilize
    /// the trait without including it.*
    fn from_ok(v: Self::Ok) -> Self;
}
```

### Initial impls

libcore will also define the following impls for the following types.

**Result**

The `Result` type includes an impl as follows:

```rust
impl<T,E> Try for Result<T, E> {
    type Ok = T;
    type Error = E;

    fn into_result(self) -> Self {
        self
    }
    
    fn from_ok(v: T) -> Self {
        Ok(v)
    }

    fn from_error(v: E) -> Self {
        Err(v)
    }
}
```

This impl permits the `?` operator to be used on results in the same
fashion as it is used today.

**Option**

The `Option` type includes an impl as follows:

```rust
mod option {
    pub struct Missing;

    impl<T> Try for Option<T>  {
        type Ok = T;
        type Error = Missing;

        fn into_result(self) -> Result<T, Missing> {
            self.ok_or(Missing)
        }
    
        fn from_ok(v: T) -> Self {
            Some(v)
        }

        fn from_error(_: Missing) -> Self {
            None
        }
    }
}    
```

Note the use of the `Missing` type, which is specific to `Option`,
rather than a generic type like `()`. This is intended to mitigate the
risk of accidental `Result -> Option` conversion. In particular, we
will only allow conversion from `Result<T, Missing>` to `Option<T>`.
The idea is that if one uses the `Missing` type as an error, that
indicates an error that can be "handled" by converting the value into
an `Option`. (This rationale was originally
[explained in a comment by Aaron Turon](https://github.com/rust-lang/rfcs/pull/1859#issuecomment-282091865).)

The use of a fresh type like `Missing` is recommended whenever one
implements `Try` for a type that does not have the `#[must_use]`
attribute (or, more semantically, that does not represent an
"unhandled error").

### Interaction with type inference

Supporting more types with the `?` operator can be somewhat limiting
for type inference. In particular, if `?` only works on values of type
`Result` (as did the old `try!` macro), then `x?` forces the type of
`x` to be `Result`. This can be significant in an expression like
`vec.iter().map(|e| ...).collect()?`, since the behavior of the
`collect()` function is determined by the type it returns. In the old
`try!` macro days, `collect()` would have been forced to return a
`Result<_, _>` -- but `?` leaves it more open.

This implies that callers of `collect()` will have to either use
`try!`, or write an explicit type annotation, something like this:

```rust
vec.iter().map(|e| ...).collect::<Result<_, _>>()?
```

Another problem (which also occurs with `try!`) stems from the use of
`From` to interconvert errors. This implies that 'nested' uses of `?`
are
[often insufficiently constrained for inference to make a decision](https://internals.rust-lang.org/t/pre-rfc-fold-ok-is-composable-internal-iteration/4434/23).
The problem here is that the nested use of `?` effectively returns
something like `From::from(From::from(err))` -- but only the starting
point (`err`) and the final type are constrained. The inner type is
not.  It's unclear how to address this problem without introducing
some form of inference fallback, which seems orthogonal from this RFC.

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

One important note is that we should publish guidelines explaining
when it is appropriate to introduce a special error type (analogous to
the `option::Missing` type included in this RFC) for use with `?`. As
expressed earlier, the rule of thumb ought to be that a special error
type should be used whenever implementing `Try` for a type that does
not, semantically, indicates an unhandled error (i.e., a type for
which the `#[must_use]` attribute would be inappropriate).

### Error messages

Another important factor is the error message when `?` is used in a
function whose return type is not suitable. The current error message
in this scenario is quite opaque and directly references the `Carrer`
trait. A better message would consider various possible cases.

**Source type does not implement Try.** If `?` is applied to a value
that does not implement the `Try` trait (for any return type), we can
give a message like

> `?` cannot be applied to a value of type `Foo`

**Return type does not implement Try.** Otherwise, if the return type
of the function does not implement `Try`, then we can report something
like this (in this case, assuming a fn that returns `()`):

> cannot use the `?` operator in a function that returns `()`

or perhaps if we want to be more strictly correct:

> `?` cannot be applied to a `Result<T, Box<Error>>` in a function that returns `()`

At this point, we could likely make a suggestion such as "consider
changing the return type to `Result<(), Box<Error>>`".

Note however that if `?` is used within an impl of a trait method, or
within `main()`, or in some other context where the user is not free
to change the type signature (modulo
[RFC 1937](https://github.com/rust-lang/rfcs/pull/1937)), then we
should not make this suggestion. In the case of an impl of a trait
defined in the current crate, we could consider suggesting that the
user change the definition of the trait.

**Errors cannot be interconverted.** Finally, if the return type `R`
does implement `Try`, but a value of type `R` cannot be constructed
from the resulting error (e.g., the function returns `Option<T>`, but
`?` is applied to a `Result<T, ()>`), then we can instead report
something like this:

> `?` cannot be applied to a `Result<T, Box<Error>>` in a function that returns `Option<T>`

This last part can be tricky, because the error can result for one of
two reasons:

- a missing `From` impl, perhaps a mistake;
- the impl of `Try` is intentionally limited, as in the case of `Option`.

We could help the user diagnose this, most likely, by offering some labels
like the following:

```rust
22 | fn foo(...) -> Option<T> {
   |                --------- requires an error of type `option::Missing`
   |     write!(foo, ...)?;
   |     ^^^^^^^^^^^^^^^^^ produces an error of type `io::Error`
   | }
```

**Consider suggesting the use of catch.** Especially in contexts
where the return type cannot be changed, but possibly in other
contexts as well, it would make sense to advise the user about how
they can catch an error instead, if they chose. Once `catch` is
stabilized, this could be as simple as saying "consider introducing a
`catch`, or changing the return type to ...". In the absence of
`catch`, we would have to suggest the introduction of a `match` block.

**Extended error message text.** In the extended error message, for
those cases where the return type cannot easily be changed, we might
consider suggesting that the fallible portion of the code is
refactored into a helper function, thus roughly following this
pattern:

```rust
fn inner_main() -> Result<(), HLError> {
    let args = parse_cmdline()?;
    // all the real work here
}

fn main() {
    process::exit(match inner_main() {
        Ok(_) => 0,
        Err(ref e) => {
            writeln!(io::stderr(), "{}", e).unwrap();
            1
        }
    });
}
```

**Implementation note:** it may be helpful for improving the error
message if `?` were not desugared when lowering from AST to HIR but
rather when lowering from HIR to MIR; however, the use of source
annotations may suffice.

# Drawbacks
[drawbacks]: #drawbacks

One drawback of supporting more types is that type inference becomes
harder. This is because an expression like `x?` no longer implies that
the type of `x` is `Result`.

There is also the risk that results or other "must use" values are
accidentally converted into other types. This is mitigated by the use
of newtypes like `option::Missing` (rather than, say, a generic type
like `()`).

# Alternatives
[alternatives]: #alternatives

### The "essentialist" approach

When this RFC was first proposed, the `Try` trait looked quite different:

```rust
trait Try<E> {
    type Success;
    fn try(self) -> Result<Self::Success, E>;
}    
```

In this version, `Try::try()` converted either to an unwrapped
"success" value, or to a error value to be propagated. This allowed
the conversion to take into account the context (i.e., one might
interconvert from a `Foo` to a `Bar` in some distinct way as one
interconverts from a `Foo` to a `Baz`).

This was changed to adopt the current "reductionist" approach, in
which all values are *first* interconverted (in a context independent
way) to an OK/Error value, and then interconverted again to match the
context using `from_error`. The reasons for the change are roughly as follows:

- The resulting trait feels simpler and more straight-forward. It also
  supports `from_ok` in a simple fashion.
- Context dependent behavior has the potential to be quite surprising.
- The use of specific types like `option::Missing` mitigates the
  primary concern that motivated the original design (avoiding overly
  loose interconversion).
- It is nice that the use of the `From` trait is now part of the `?` desugaring,
  and hence supported universally across all types.
- The interaction with the orphan rules is made somewhat nicer. For example,
  using the essentialist alternative, one might like to have a trait
  that permits a `Result` to be returned in a function that yields `Poll`.
  That would require an impl like this `impl<T,E> Try<Poll<T,E>> for Result<T, E>`,
  but this impl runs afoul of the orphan rules.

### Traits implemented over higher-kinded types

The desire to avoid "free interconversion" between `Result` and
`Option` seemed to suggest that the `Carrier` trait ought to be
defined over higher-kinded types (or generic associated types) in some
form. The most obvious downside of such a design is that Rust does not
offer higher-kinded types nor anything equivalent to them today, and
hence we would have to block on that design effort. But it also turns
out that HKT is
[not a particularly good fit for the problem](https://github.com/rust-lang/rust/pull/35056#issuecomment-240129923). To
start, consider what "kind" the `Self` parameter on the `Try` trait
would have to have.  If we were to implement `Try` on `Option`, it
would presumably then have kind `type -> type`, but we also wish to
implement `Try` on `Result`, which has kind `type -> type ->
type`. There has even been talk of implementing `Try` for simple types
like `bool`, which simply have kind `type`. More generally, the
problems encountered are quite similar to the problems that
[Simon Peyton-Jones describes in attempting to model collections using HKT](https://www.microsoft.com/en-us/research/wp-content/uploads/1997/01/multi.pdf):
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
`Deref` and not `Star` or `Asterisk`), it was more appropriate to name
the trait `Try`, which seems to be the best name for the operation in
question.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
