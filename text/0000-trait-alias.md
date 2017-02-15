- Feature Name: Trait alias
- Start Date: 2016-08-31
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Traits can be aliased the same way types can be aliased with the `type` keyword.

# Motivation
[motivation]: #motivation

## First motivation: `impl`

Sometimes, some traits are defined with parameters. For instance:

```rust
pub trait Foo<T> {
  // ...
}
```

It’s not uncommon to do that in *generic* crates and implement them in *backend* crates, where the
`T` template parameter gets substituted with a *backend* type.

```rust
// in the backend crate
pub struct Backend;

impl trait Foo<Backend> for i32 {
  // ...
}
```

Users who want to use that crate will have to export both the trait `Foo` from the generic crate
*and* the backend singleton type from the backend crate. Instead, we would like to be able to let
the backend singleton type hidden in the crate. The first shot would be to create a new trait for
our backend:

```rust
pub trait FooBackend: Foo<Backend> {
  // ...
}

fn use_foo<A>(_: A) where A: FooBackend {}
```

If you try to pass an object that implements `Foo<Backend>`, that won’t work, because it doesn’t
implement `FooBackend`. However, we can make it work with the following universal `impl`:

```rust
impl<T> FooBackend for T where T: Foo<Backend> {}
```

With that, it’s now possible to pass an object that implements `Foo<Backend>` to a function
expecting a `FooBackend`. However, what about impl blocks? What happens if we implement only
`FooBackend`? Well, we cannot, because the trait explicitely states that we need to implement
`Foo<Backend>`. We hit a problem here. The problem is that even though there’s a compatibility at
the `trait bound` level between `Foo<Backend>` and `FooBackend`, there’s none at the `impl` level,
so all we’re left with is implementing `Foo<Backend>` – that will also provide an implementation for
`FooBackend` because of the universal implementation just above.

## Second example: ergonomic collections and scrapping boilerplate

Another example is associated types. Take the following [trait from tokio](https://docs.rs/tokio-service/0.1.0/tokio_service/trait.Service.html):

```rust
pub trait Service {
  type Request;
  type Response;
  type Error;
  type Future: Future<Item=Self::Response, Error=Self::Error>;
  fn call(&self, req: Self::Request) -> Self::Future;
}
```

It would be nice to be able to create a few aliases to remove boilerplate for very common
combinations of associated types with `Service`.

```rust
Service<Request = http::Request, Response = http::Response, Error = http::Error>;
```

The trait above is a http service trait which only the associated type `Future` is left to be
implemented. Such an alias would be very appealing because it would remove copying the whole
`Service` trait into use sites – trait bounds, or even trait impls. Scrapping such an annoying
boilerplate is a definitive plus to the language and might be one of the most interesting use case.

# Detailed design
[design]: #detailed-design

The syntax chosen to make a *trait alias* is:

```rust
trait TraitAlias = Trait;
```
It creates a new trait alias `TraitAlias` that will resolve to `Trait`.

```rust
trait DebugAlias = Debug;
```

Trait aliasing to combinations of traits is also provided with the standard `+` construct:

```rust
trait DebugDefault = Debug + Default;
```

Optionally, if needed, one can provide a `where` clause to express *bounds*:

```rust
trait DebugDefault = Debug where Self: Default; // same as the example above
```

Furthermore, it’s possible to use only the `where` clause by using the special syntax `= _`:

```rust
trait DebugDefault = _ where Self: Debug + Default;
```

Trait aliases can be used in any place arbitrary bounds would be syntactically legal. However, you
cannot use them in `impl` place but can have them as *trait objects*, in *where-clauses* and *type
parameters declarations* of course.

# Drawbacks
[drawbacks]: #drawbacks

The syntax `trait TraitAlias = Trait` makes parsers need a lookhead.

# Alternatives
[alternatives]: #alternatives

It’s possible to create a new trait that derives the trait to alias, and provide a universal `impl`:

```rust
trait Foo {}

trait FooFakeAlias: Foo {}

impl<T> Foo for T where T: FooFakeAlias {}
```

This works for trait objects and trait bounds only. You cannot implement `FooFakeAlias` directly
because you need to implement `Foo` first – hence, you don’t really need `FooFakeAlias` if you can
implement `Foo`.

There’s currently no alternative to the impl problem described here.

# Unresolved questions
[unresolved]: #unresolved-questions

What about bounds on type variable declaration in the trait alias? Consider the following:

```rust
trait Foo<T: Bar> = PartialEq<T>
```

`PartialEq` has no super-trait `Bar`, but we’re adding one via our trait alias. What is the behavior
of such a feature? We can desugar it to:

```rust
trait Foo<T> = _ where Self: PartialEq<T>, T: Bar
```

This semantic has to be discussed a bit more.
