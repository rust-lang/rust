- Feature Name: Trait alias
- Start Date: 2016-08-31
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Traits can be aliased with the `trait TraitAlias = …;` construct. Currently, the right hand side is
a bound – a single trait, a combination with `+` traits and lifetimes. Type parameters and
lifetimes can be added to the *trait alias* if needed.

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

## Syntax

The syntax chosen to make a *trait alias* is:

```rust
trait TraitAlias = Trait;
```

Trait aliasing to combinations of traits is also provided with the standard `+` construct:

```rust
trait DebugDefault = Debug + Default;
```

Optionally, if needed, one can provide a `where` clause to express *bounds*:

```rust
trait DebugDefault = Debug where Self: Default; // same as the example above
```

Furthermore, it’s possible to use only the `where` clause by leaving the list of traits empty:

```rust
trait DebugDefault = where Self: Debug + Default;
```

Specifically, the grammar being added is, in informal notation:

```
ATTRIBUTE* VISIBILITY? trait IDENTIFIER(<GENERIC_PARAMS>)? = GENERIC_BOUNDS (where PREDICATES)?;
```

`GENERIC_BOUNDS` is a list of zero or more traits and lifetimes separated by `+`, the same as the
current syntax for bounds on a type parameter, and `PREDICATES` is a comma-separated list of zero or
more predicates, just like any other `where` clause. A trait alias containing only lifetimes (`trait
Static = 'static;`) is not allowed.

## Semantics

Trait aliases can be used in any place arbitrary bounds would be syntactically legal.

You cannot directly `impl` a trait alias, but can have them as *bounds*, *trait objects* and *impl
Trait*.

When using a trait alias as an object type, it is subject to object safety restrictions _after_
substituting the aliased traits. This means:

1. It contains an object safe trait, optionally a lifetime, and zero or more of these other bounds:
   `Send`, `Sync` (that is, `trait Show = Display + Debug;` would not be object safe).
2. All the associated types of the trait need to be specified.
3. The `where` clause, if present, only contains bounds on `Self`.

Some examples:

```rust
trait Sink = Sync;
trait ShareableIterator = Iterator + Sync;
trait PrintableIterator = Iterator<Item=i32> + Display;
trait IntIterator = Iterator<Item=i32>;

fn foo1<T: ShareableIterator>(...) { ... } // ok
fn foo2<T: ShareableIterator<Item=i32>>(...) { ... } // ok
fn bar1(x: Box<ShareableIterator>) { ... } // ERROR: associated type not specified
fn bar2(x: Box<ShareableIterator<Item=i32>>) { ... } // ok
fn bar3(x: Box<PrintableIterator>) { ... } // ERROR: too many traits (*)
fn bar4(x: Box<IntIterator + Sink + 'static>) { ... } // ok (*)
```

The lines marked with `(*)` assume that [#24010](https://github.com/rust-lang/rust/issues/24010) is
fixed.

# Teaching
[teaching]: #teaching

[Traits](https://doc.rust-lang.org/book/traits.html) are obviously a huge prerequisite. Traits
aliases could be introduced at the end of that chapter.

Conceptually, a *trait alias* is a syntax shortcut used to reason about one or more trait(s).
Inherently, the *trait alias* is usable in a limited set of places:

- as a *bound*: exactly like a *trait*, a *trait alias* can be used to constraint a type (type
  parameters list, where-clause)
- as a *trait object*: same thing as with a *trait*, a *trait alias* can be used as a *trait object*
  if it fits object safety restrictions (see above in the [semantics](#semantics) section)
- in an [`impl Trait`](https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md)

Examples should be showed for all of the three cases above:

#### As a bound

```rust
trait StringIterator = Iterator<Item=String>;

fn iterate<SI>(si: SI) where SI: StringIterator {} // used as bound
```

#### As a trait object

```rust
fn iterate_object(si: &StringIterator) {} // used as trait object
```

#### In an `impl Trait`

```rust
fn string_iterator_debug() -> impl Debug + StringIterator {} // used in an impl Trait
```

As shown above, a *trait alias* can substitute associated types. It doesn’t have to substitute them
all. In that case, the *trait alias* is left incomplete and you have to pass it the associated types
that are left. Example with the [tokio case](#second-example-ergonomic-collections-and-scrapping-boilerplate):

```rust
pub trait Service {
  type Request;
  type Response;
  type Error;
  type Future: Future<Item=Self::Response, Error=Self::Error>;
  fn call(&self, req: Self::Request) -> Self::Future;
}

trait HttpService = Service<Request = http::Request, Response = http::Response, Error = http::Error>;

trait MyHttpService = HttpService<Future = MyFuture>; // assume MyFuture exists and fulfills the rules to be used in here
```

# Drawbacks
[drawbacks]: #drawbacks

- Adds another construct to the language.

- The syntax `trait TraitAlias = Trait` requires lookahead in the parser to disambiguate a trait
  from a trait alias.

# Alternatives
[alternatives]: #alternatives

## Should we use `type` as the keyword instead of `trait`?

`type Foo = Bar;` already creates an alias `Foo` that can be used as a trait object.

If we used `type` for the keyword, this would imply that `Foo` could also be used as a bound as
well. If we use `trait` as proposed in the body of the RFC, then `type Foo = Bar;` and
`trait Foo = Bar;` _both_ create an alias for the object type, but only the latter creates an alias
that can be used as a bound, which is a confusing bit of redundancy.

However, this mixes the concepts of types and traits, which are different, and allows nonsense like
`type Foo = Rc<i32> + f32;` to parse.

## Supertraits & universal `impl`

It’s possible to create a new trait that derives the trait to alias, and provide a universal `impl`

```rust
trait Foo {}

trait FooFakeAlias: Foo {}

impl<T> Foo for T where T: FooFakeAlias {}
```

This works for trait objects and trait bounds only. You cannot implement `FooFakeAlias` directly
because you need to implement `Foo` first – hence, you don’t really need `FooFakeAlias` if you can
implement `Foo`.

There’s currently no alternative to the impl problem described here.

## `ContraintKinds`

Similar to Haskell's ContraintKinds, we could declare an entire predicate as a reified list of
constraints, instead of creating an alias for a set of supertraits and predicates. Syntax would be
something like `constraint Foo<T> = T: Bar, Vec<T>: Baz;`, used as `fn quux<T>(...) where Foo<T> { ... }`
(i.e. direct substitution). Trait object usage is unclear.

# Unresolved questions
[unresolved]: #unresolved-questions
 
## Which bounds need to be repeated when using a trait alias?

[RFC 1927](https://github.com/rust-lang/rfcs/pull/1927) intends to change the rules here for traits,
and we likely want to have the rules for trait aliases be the same to avoid confusion.

The `constraint` alternative sidesteps this issue.

## What about bounds on type variable declaration in the trait alias?

```rust
trait Foo<T: Bar> = PartialEq<T>;
```

`PartialEq` has no super-trait `Bar`, but we’re adding one via our trait alias. What is the behavior
of such a feature? One possible desugaring is:

```rust
trait Foo<T> = where Self: PartialEq<T>, T: Bar;
```

[Issue 21903](https://github.com/rust-lang/rust/issues/21903) explains the same problem for type
aliasing.
