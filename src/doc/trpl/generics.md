% Generics

Sometimes, when writing a function or data type, we may want it to work for
multiple types of arguments. For example, remember our `OptionalInt` type?

```{rust}
enum OptionalInt {
    Value(int),
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
let x: Option<int> = Some(5i);
```

In the type declaration, we say `Option<int>`. Note how similar this looks to
`Option<T>`. So, in this particular `Option`, `T` has the value of `int`. On
the right-hand side of the binding, we do make a `Some(T)`, where `T` is `5i`.
Since that's an `int`, the two sides match, and Rust is happy. If they didn't
match, we'd get an error:

```{rust,ignore}
let x: Option<f64> = Some(5i);
// error: mismatched types: expected `core::option::Option<f64>`
// but found `core::option::Option<int>` (expected f64 but found int)
```

That doesn't mean we can't make `Option<T>`s that hold an `f64`! They just have to
match up:

```{rust}
let x: Option<int> = Some(5i);
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
enum Result<H, N> {
    Ok(H),
    Err(N),
}
```

if we wanted to. Convention says that the first generic parameter should be
`T`, for 'type,' and that we use `E` for 'error.' Rust doesn't care, however.

The `Result<T, E>` type is intended to be used to return the result of a
computation, and to have the ability to return an error if it didn't work out.
Here's an example:

```{rust}
let x: Result<f64, String> = Ok(2.3f64);
let y: Result<f64, String> = Err("There was an error.".to_string());
```

This particular Result will return an `f64` if there's a success, and a
`String` if there's a failure. Let's write a function that uses `Result<T, E>`:

```{rust}
fn inverse(x: f64) -> Result<f64, String> {
    if x == 0.0f64 { return Err("x cannot be zero!".to_string()); }

    Ok(1.0f64 / x)
}
```

We don't want to take the inverse of zero, so we check to make sure that we
weren't passed zero. If we were, then we return an `Err`, with a message. If
it's okay, we return an `Ok`, with the answer.

Why does this matter? Well, remember how `match` does exhaustive matches?
Here's how this function gets used:

```{rust}
# fn inverse(x: f64) -> Result<f64, String> {
#     if x == 0.0f64 { return Err("x cannot be zero!".to_string()); }
#     Ok(1.0f64 / x)
# }
let x = inverse(25.0f64);

match x {
    Ok(x) => println!("The inverse of 25 is {}", x),
    Err(msg) => println!("Error: {}", msg),
}
```

The `match` enforces that we handle the `Err` case. In addition, because the
answer is wrapped up in an `Ok`, we can't just use the result without doing
the match:

```{rust,ignore}
let x = inverse(25.0f64);
println!("{}", x + 2.0f64); // error: binary operation `+` cannot be applied
           // to type `core::result::Result<f64,collections::string::String>`
```

This function is great, but there's one other problem: it only works for 64 bit
floating point values. What if we wanted to handle 32 bit floating point as
well? We'd have to write this:

```{rust}
fn inverse32(x: f32) -> Result<f32, String> {
    if x == 0.0f32 { return Err("x cannot be zero!".to_string()); }

    Ok(1.0f32 / x)
}
```

Bummer. What we need is a *generic function*. Luckily, we can write one!
However, it won't _quite_ work yet. Before we get into that, let's talk syntax.
A generic version of `inverse` would look something like this:

```{rust,ignore}
fn inverse<T>(x: T) -> Result<T, String> {
    if x == 0.0 { return Err("x cannot be zero!".to_string()); }

    Ok(1.0 / x)
}
```

Just like how we had `Option<T>`, we use a similar syntax for `inverse<T>`.
We can then use `T` inside the rest of the signature: `x` has type `T`, and half
of the `Result` has type `T`. However, if we try to compile that example, we'll get
an error:

```text
error: binary operation `==` cannot be applied to type `T`
```

Because `T` can be _any_ type, it may be a type that doesn't implement `==`,
and therefore, the first line would be wrong. What do we do?

To fix this example, we need to learn about another Rust feature: traits.
