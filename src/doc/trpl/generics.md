% Generics

Sometimes, when writing a function or data type, we may want it to work for
multiple types of arguments. For example, remember our `OptionalInt` type?

```{rust}
enum OptionalInt {
    Value(i32),
    Missing,
}
```

If we wanted to also have an `OptionalFloat64`, we would need a new enum:

```{rust}
enum OptionalFloat64 {
    Valuef64(f64),
    Missingf64,
}
```

This is really unfortunate. Luckily, Rust has a feature that gives us a better
way: generics. Generics are called *parametric polymorphism* in type theory,
which means that they are types or functions that have multiple forms (*poly*
is multiple, *morph* is form) over a given parameter (*parametric*).

Anyway, enough with type theory declarations, let's check out the generic form
of `OptionalInt`. It is actually provided by Rust itself, and looks like this:

```rust
enum Option<T> {
    Some(T),
    None,
}
```

The `<T>` part, which you've seen a few times before, indicates that this is
a generic data type. Inside the declaration of our enum, wherever we see a `T`,
we substitute that type for the same type used in the generic. Here's an
example of using `Option<T>`, with some extra type annotations:

```{rust}
let x: Option<i32> = Some(5);
```

In the type declaration, we say `Option<i32>`. Note how similar this looks to
`Option<T>`. So, in this particular `Option`, `T` has the value of `i32`. On
the right-hand side of the binding, we do make a `Some(T)`, where `T` is `5`.
Since that's an `i32`, the two sides match, and Rust is happy. If they didn't
match, we'd get an error:

```{rust,ignore}
let x: Option<f64> = Some(5);
// error: mismatched types: expected `core::option::Option<f64>`,
// found `core::option::Option<_>` (expected f64 but found integral variable)
```

That doesn't mean we can't make `Option<T>`s that hold an `f64`! They just have to
match up:

```{rust}
let x: Option<i32> = Some(5);
let y: Option<f64> = Some(5.0f64);
```

This is just fine. One definition, multiple uses.

Generics don't have to only be generic over one type. Consider Rust's built-in
`Result<T, E>` type:

```{rust}
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

This type is generic over _two_ types: `T` and `E`. By the way, the capital letters
can be any letter you'd like. We could define `Result<T, E>` as:

```{rust}
enum Result<A, Z> {
    Ok(A),
    Err(Z),
}
```

if we wanted to. Convention says that the first generic parameter should be
`T`, for 'type,' and that we use `E` for 'error.' Rust doesn't care, however.

The `Result<T, E>` type is intended to be used to return the result of a
computation, and to have the ability to return an error if it didn't work out.
