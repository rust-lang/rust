- Start Date: 2015-01-22
- RFC PR: [rust-lang/rfcs#587](https://github.com/rust-lang/rfcs/pull/587)
- Rust Issue: [rust-lang/rust#21527](https://github.com/rust-lang/rust/issues/21527)

# Summary

The `Fn` traits should be modified to make the return type an associated type.

# Motivation

The strongest reason is because it would permit impls like the following
(example from @alexcrichton):

```rust
impl<R,F> Foo for F : FnMut() -> R { ... }
```

This impl is currently illegal because the parameter `R` is not
constrained. (This also has an impact on my attempts to add variance,
which would require a "phantom data" annotation for `R` for the same
reason; but that RFC is not quite ready yet.)

Another related reason is that it often permits fewer type parameters.
Rather than having a distinct type parameter for the return type, the
associated type projection `F::Output` can be used. Consider the standard
library `Map` type:

```rust
struct Map<A,B,I,F>
    where I : Iterator<Item=A>,
          F : FnMut(A) -> B,
{
    ...
}

impl<A,B,I,F> Iterator for Map<A,B,I,F>
    where I : Iterator<Item=A>,
          F : FnMut(A) -> B,
{
    type Item = B;
    ...
}
```

This type could be equivalently written:

```rust
struct Map<I,F>
    where I : Iterator, F : FnMut<(I::Item,)>
{
    ...
}

impl<I,F> Iterator for Map<I,F>,
    where I : Iterator,
          F : FnMut<(I::Item,)>,
{
    type Item = F::Output;
    ...
}
```

This example highlights one subtle point about the `()` notation,
which is covered below.

# Detailed design

The design has been implemented. You can see it in [this pull
request]. The `Fn` trait is modified to read as follows:

```rust
trait Fn<A> {
    type Output;
    fn call(&self, args: A) -> Self::Output;
}
```

The other traits are modified in an analogous fashion.

[this pull request]: https://github.com/rust-lang/rust/pull/21019

### Parentheses notation

The shorthand `Foo(...)` expands to `Foo<(...), Output=()>`. The
shorthand `Foo(..) -> B` expands to `Foo<(...), Output=B>`. This
implies that if you use the parenthetical notation, you must supply a
return type (which could be a new type parameter). If you would prefer
to leave the return type unspecified, you must use angle-bracket
notation. (Note that using angle-bracket notation with the `Fn` traits
is currently feature-gated, as [described here][18875].)

[18875]: https://github.com/rust-lang/rust/issues/18875

This can be seen in the In the `Map` example from the
introduction. There the `<>` notation was used so that `F::Output` is
left unbound:

```rust
struct Map<I,F>
    where I : Iterator, F : FnMut<(I::Item,)>
```

An alternative would be to retain the type parameter `B`:

```rust
struct Map<B,I,F>
    where I : Iterator, F : FnMut(I::Item) -> B
```

Or to remove the bound on `F` from the type definition and use it only in the impl:

```rust
struct Map<I,F>
    where I : Iterator
{
    ...
}

impl<B,I,F> Iterator for Map<I,F>,
    where I : Iterator,
          F : FnMut(I::Item) -> B
{
    type Item = F::Output;
    ...
}
```

Note that this final option is not legal without this change, because
the type parameter `B` on the impl woudl be unconstrained.

# Drawbacks

### Cannot overload based on return type alone

This change means that you cannot overload indexing to "model" a trait
like `Default`:

```rust
trait Default {
    fn default() -> Self;
}
```

That is, I can't do something like the following:

```rust
struct Defaulty;
impl<T:Default> Fn<()> for Defaulty {
    type Output = T;

    fn call(&self) -> T {
        Default::default()
    }
}
```

This is not possible because the impl type parameter `T` is not constrained.

This does not seem like a particularly strong limitation. Overloaded
call notation is already less general than full traits in various ways
(for example, it lacks the ability to define a closure that always
panics; that is, the `!` notation is not a type and hence something
like `FnMut() -> !` is not legal). The ability to overload based on return type
is not removed, it is simply not something you can model using overloaded operators.

# Alternatives

### Special syntax to represent the lack of an `Output` binding

Rather than having people use angle-brackets to omit the `Output`
binding, we could introduce some special syntax for this purpose.  For
example, `FnMut() -> ?` could desugar to `FnMut<()>` (whereas
`FnMut()` alone desugars to `FnMut<(), Output=()>`). The first
suggestion that is commonly made is `FnMut() -> _`, but that has an
existing meaning in a function context (where `_` represents a fresh
type variable).

### Change meaning of `FnMut()` to not bind the output

We could make `FnMut()` desugar to `FnMut<()>`, and hence require an
explicit `FnMut() -> ()` to bind the return type to unit.  This feels
suprising and inconsistent.


