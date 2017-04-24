- Feature Name: Trait alias
- Start Date: 2016-08-31
- RFC PR: [rust-lang/rfcs#1733](https://github.com/rust-lang/rfcs/pull/1733)
- Rust Issue: [rust-lang/rust#41517](https://github.com/rust-lang/rust/issues/41517)

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

impl Foo<Backend> for i32 {
  // ...
}
```

Users who want to use that crate will have to export both the trait `Foo` from the generic crate
*and* the backend singleton type from the backend crate. Instead, we would like to be able to leave
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

### Declaration

The syntax chosen to declare a *trait alias* is:

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

It’s also possible to partially bind associated types of the right hand side:

```rust
trait IntoIntIterator = IntoIterator<Item=i32>;
```

This would leave `IntoIntIterator` with a *free parameter* being `IntoIter`, and it should be bind
the same way associated types are bound with regular traits:

```rust
fn foo<I>(int_iter: I) where I: IntoIntIterator<IntoIter = std::slice::Iter<i32>> {}
```

A trait alias can be parameterized over types and lifetimes, just like traits themselves:

```rust
trait LifetimeParametric<'a> = Iterator<Item=Cow<'a, [i32]>>;`

trait TypeParametric<T> = Iterator<Item=Cow<'static, [T]>>;
```

---

Specifically, the grammar being added is, in informal notation:

```
ATTRIBUTE* VISIBILITY? trait IDENTIFIER(<GENERIC_PARAMS>)? = GENERIC_BOUNDS (where PREDICATES)?;
```

`GENERIC_BOUNDS` is a list of zero or more traits and lifetimes separated by `+`, the same as the
current syntax for bounds on a type parameter, and `PREDICATES` is a comma-separated list of zero or
more predicates, just like any other `where` clause.
`GENERIC_PARAMS` is a comma-separated list of zero or more lifetime and type parameters,
with optional bounds, just like other generic definitions.

## Use semantics

You cannot directly `impl` a trait alias, but you can have them as *bounds*, *trait objects* and
*impl Trait*.

----

It is an error to attempt to override a previously specified
equivalence constraint with a non-equivalent type. For example:

```rust
trait SharableIterator = Iterator + Sync;
trait IntIterator = Iterator<Item=i32>;

fn quux1<T: SharableIterator<Item=f64>>(...) { ... } // ok
fn quux2<T: IntIterator<Item=i32>>(...) { ... } // ok (perhaps subject to lint warning)
fn quux3<T: IntIterator<Item=f64>>(...) { ... } // ERROR: `Item` already constrained

trait FloIterator = IntIterator<Item=f64>; // ERROR: `Item` already constrained
```

---

When using a trait alias as a trait object, it is subject to object safety restrictions *after*
substituting the aliased traits. This means:

1. it contains an object safe trait, optionally a lifetime, and zero or more of these other bounds:
   `Send`, `Sync` (that is, `trait Show = Display + Debug;` would not be object safe);
2. all the associated types of the trait need to be specified;
3. the `where` clause, if present, only contains bounds on `Self`.

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

### Ambiguous constraints

If there are multiple associated types with the same name in a trait alias,
then it is a static error ("ambiguous associated type") to attempt to
constrain that associated type via the trait alias. For example:

```rust
trait Foo { type Assoc; }
trait Bar { type Assoc; } // same name!

// This works:
trait FooBar1 = Foo<Assoc = String> + Bar<Assoc = i32>;

// This does not work:
trait FooBar2 = Foo + Bar;
fn badness<T: FooBar2<Assoc = String>>() { } // ERROR: ambiguous associated type

// Here are ways to workaround the above error:
fn better1<T: FooBar2 + Foo<Assoc = String>>() { } // (leaves Bar::Assoc unconstrained)
fn better2<T: FooBar2 + Foo<Assoc = String> + Bar<Assoc = i32>>() { } // constrains both
```

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

## `ConstraintKinds`

Similar to GHC’s `ContraintKinds`, we could declare an entire predicate as a reified list of
constraints, instead of creating an alias for a set of supertraits and predicates. Syntax would be
something like `constraint Foo<T> = T: Bar, Vec<T>: Baz;`, used as `fn quux<T>(...) where Foo<T> { ... }`
(i.e. direct substitution). Trait object usage is unclear.

## Syntax for sole `where` clause.

The current RFC specifies that it is possible to use only the `where` clause by leaving the list of traits empty:

```rust
trait DebugDefault = where Self: Debug + Default;
```

This is one of many syntaxes that are available for this construct. Alternatives include:

 * `trait DebugDefault where Self: Debug + Default;` (which has been [considered and discarded](https://github.com/rust-lang/rfcs/pull/1733#issuecomment-257993316) because [it might look](https://github.com/rust-lang/rfcs/pull/1733#issuecomment-258495468) too much like a new trait definition)
 * `trait DebugDefault = _ where Self: Debug + Default;` (which was [considered and then removed](https://github.com/rust-lang/rfcs/pull/1733/commits/88d3074957276c7201147fc625f18e0ebcecc1b9#diff-ae27a1a8d977f731e67823349151bed5L116) because it is [technically unnecessary](https://github.com/rust-lang/rfcs/pull/1733#issuecomment-284252196))
 * `trait DebugDefault = Self where Self: Debug + Default;` (analogous to previous case but not formally discussed)

# Unresolved questions
[unresolved]: #unresolved-questions
 
## Trait alias containing only lifetimes

This is annoying. Consider:

```rust
trait Static = 'static;

fn foo<T>(t: T) where T: Static {}
```

Such an alias is legit. However, I feel concerned about the actual meaning of the declaration – i.e.
using the `trait` keyword to define alias on *lifetimes* seems a wrong design choice and seems not
very consistent.

If we chose another keyword, like `constraint`, I feel less concerned and it would open further
opportunies – see the `ConstraintKinds` alternative discussion above.

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

**Note: what about the following proposal below?**

When using a trait alias as a bound, you cannot add extra bound on the input parameters, like in the
following:

```rust
trait Foo<T: Bar> = PartialEq<T>;
```

Here, `T` adds a `Bar` bound. Now consider:

```rust
trait Bar<T> = PartialEq<T: Bar>;
```

Currently, we don’t have a proper understanding of that situation, because we’re adding in both
cases a bound, and we don’t know how to disambiguate between *pre-condition* and *implication*. That
is, is that added `Bar` bound a constraint that `T` must fulfil in order for the trait alias to be
met, or is it a constraint the trait alias itself adds? To disambiguate, consider:

```rust
trait BarPrecond<T> where T: Bar = PartialEq<T>;
trait BarImplic<T> = PartialEq<T> where T: Bar;
trait BarImpossible<T> where T: Bar = PartialEq<T> where T: Bar;
```

`BarPrecond` would require the use-site code to fulfil the constraint, like the following:

```rust
fn foo<A, T>() where A: BarPrecond<T>, T: Bar {}
```

`BarImplic` would give us `T: Bar`:

```rust
fn foo<A, T>() where A: BarImplic<T> {
  // T: Bar because given by BarImplic<T>
}
```

`BarImpossible` wouldn’t compile because we try to express a pre-condition and an implication for
the same bound at the same time. However, it’d be possible to have both a pre-condition and an
implication on a parameter:

```rust
trait BarBoth<T> where T: Bar = PartialEq<T> where T: Debug;

fn foo<A, T>() where A: BarBoth<T>, T: Bar {
  // T: Debug because given by BarBoth
}
```
