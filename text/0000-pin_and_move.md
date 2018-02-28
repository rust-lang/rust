- Feature Name: pin_and_move
- Start Date: 2018-02-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Introduce new APIs to libcore / libstd to serve as safe abstractions for data
which cannot be safely moved around.

# Motivation
[motivation]: #motivation

A longstanding problem for Rust has been dealing with types that should not be
moved. A common motivation for this is when a struct contains a pointer into
its own representation - moving that struct would invalidate that pointer. This
use case has become especially important recently with work on generators.
Because generators essentially reify a stackframe into an object that can be
manipulated in code, it is likely for idiomatic usage of a generator to result
in such a self-referential type, if it is allowed.

This proposal adds an API to std which would allow you to guarantee that a
particular value will never move again, enabling safe APIs that rely on
self-references to exist.

# Guide-level explanation

The core goal of this RFC is to **provide a reference type where the referent is guaranteed to never move before being dropped**. We want to do this with a minimum disruption to the type system, and in fact, this RFC shows that we can achieve the goal without *any* type system changes.

Let's take that goal apart, piece by piece, from the perspective of the futures (i.e. async/await) use case:

- **Reference type**. The reason we need a reference type is that, when working with things like futures, we generally want to combine smaller futures into larger ones, and only at the top level put an entire resulting future into some immovable location. Thus, we need a reference type for methods like `poll`, so that we can break apart a large future into its smaller components, while retaining the guarantee about immobility.

- **Never to move before being dropped**. Again looking at the futures case, once we being `poll`ing a future, we want it to be able to store references into itself, which is possible if we can guarantee that the whole future will never move. We don't try to track *whether* such references exist at the type level, since that would involve cumbersome typestate; instead, we simply decree that by the time you initially `poll`, you promise to never move an immobile future again.

At the same time, we want to support futures (and iterators, etc.) that *can* move. While it's possible to do so by providing two distinct `Future` (or `Iterator`, etc) traits, such designs incur unacceptable ergonomic costs.

The key insight of this RFC is that we can create a new library type, `Pin<'a, T>`, which encompasses *both* moveable and immobile referents. The type is paired with a new auto trait, `Move`, which determines the meaning of `Pin<'a, T>`:

- If `T: Move` (which is the default), then `Pin<'a, T>` is entirely equivalent to `&'a mut T`.
- If `T: !Move`, then `Pin<'a, T>` provides a unique reference to a `T` with lifetime `'a`, but only provides `&'a T` access safely. It also guarantees that the referent will *never* be moved. However, getting `&'a mut T` access is unsafe, because operations like `mem::replace` mean that `&mut` access is enough to move data out of the referent; you must promise not to do so.

To be clear: the *sole* function of `Move` is to control the meaning of `Pin`. Making `Move` an auto trait means that the vast majority of types are automatically "movable", so `Pin` degenerates to `&mut`. In the case that you need immobility, you *opt out* of `Move`, and then `Pin` becomes meaningful for your type.

Putting this all together, we arrive at the following definition of `Future`:

```rust
trait Future {
    type Item;
    type Error;

    fn poll(self: Pin<Self>, cx: task::Context) -> Poll<Self::Item, Self::Error>;
}
```

By default when implementing `Future` for a struct, this definition is equivalent to today's, which takes `&mut self`. But if you want to allow self-referencing in your future, you just opt out of `Move`, and `Pin` takes care of the rest.

The final piece of this RFC is the `Anchor` type, which is just a `Box` that doesn't allow moving out, but *does* allow you to acquire a `Pin` reference.

# Reference-level explanation

## The `Move` auto trait

This new auto trait is added to the `core::marker` and `std::marker` modules:

```rust
pub unsafe auto trait Move { }
```

A type implements `Move` if in its stack representation, it does not contain
internal references to other positions within its stack representation. Nearly
every type in Rust is `Move`.

This trait is a lang item, but only to generate negative impls for certain
generators. Unlike previous `?Move` proposals, and unlike some traits like
`Sized` and `Copy`, this trait does not impose any compiler-based semantics
types that do or don't implement it. Instead, the semantics are entirely
enforced through library APIs which use `Move` as a marker.

## The `Anchor` type

An anchor is a new kind of smart pointer. It is very much like a box - it is a
heap-allocated, exclusive-ownership type - but it provides additional
constraints. Unless the type it references implements `Move`, it is not
possible to mutably dereference the `Anchor` or move out of it in safe code. It does
implement immutable Deref for all types, including types that don't implement
`Move`.

```rust
#[fundamental]
struct Anchor<T: ?Sized> {
    inner: Box<T>,
}

impl<T> Anchor<T> {
     pub fn new(data: T) -> Anchor<T>;
}

impl<T: ?Sized> Anchor<T> {
     pub fn as_pin<'a>(&'a mut self) -> Pin<'a, T>;

     pub unsafe fn get_mut(this: &mut Anchor<T>) -> &mut T;

     pub unsafe fn into_inner_unchecked(this: Anchor<T>) -> T;
}

impl<T: Move + ?Sized> Anchor<T> {
     pub unsafe fn into_inner(this: Anchor<T>) -> T;
}

impl<T: ?Sized> Deref for Anchor<T> {
     type Target = T;
}

impl<T: Move + ?Sized> DerefMut for Anchor<T> { }

unsafe impl<T: ?Sized> Move for Anchor<T> { }
```

For types that do not implement `Move`, instead of using mutable references,
users of `Anchor` will use the `Pin` type, described in the next section.

## The `Pin` type

The `Pin` type is a wrapper around a mutable reference. If the type it
references is `!Move`, the `Pin` type guarantees that the referenced data will
never be moved again.

```rust
#[fundamental]
struct Pin<'a, T: ?Sized + 'a> {
    data: &'a mut T,
}

impl<'a, T: ?Sized + Move> Pin<'a, T> {
    pub fn new(data: &'a mut T) -> Pin<'a, T>;
}

impl<'a, T: ?Sized> Pin<'a, T> {
    pub unsafe fn new_unchecked(data: &'a mut T) -> Pin<'a, T>;

    pub unsafe fn get_mut(this: Pin<'a, T>) -> &'a mut T;

    pub fn borrow<'b>(this: &'b mut Pin<'a, T>) -> Pin<'b, T>;
}

impl<'a, T: ?Sized> Deref for Pin<'a, T> {
    type Target = T;
}

impl<'a, T: ?Sized + Move> DerefMut for Pin<'a, T> { }
```

For types which implement `Move`, `Pin` is essentially the same as an `&mut T`.
But for types which do not, the conversion between `&mut T` and `Pin` is
unsafe.

The contract on the unsafe part of `Pin`s API is that a Pin cannot be
constructed if the data it references would ever move again, and that it cannot
be converted into a mutable reference if the data might ever be moved out of
that reference. In other words, if you have a `Pin` containing data which does
not implement `Move`, you have a guarantee that that data will never move.

The `Anchor` type has a method `as_pin`, which returns a `Pin`, because it
upholds the guarantees necessary to safely construct a `Pin`.

## Immovable generators

Today, the unstable generators feature has an option to create generators which
contain references that live across yield points - these are, in effect,
internal references into the generator's state machine. Because internal
references are invalidated if the type is moved, these kinds of generators
("immovable generators") are currently unsafe to create.

Once the arbitrary_self_types feature becomes object safe, we will make three
changes to the generator API:

1. We will change the `resume` method to take self by `self: Pin<Self>` instead
   of `&mut self`.
2. We will implement `!Move` for the anonymous type of an immovable generator.
3. We will make it safe to define an immovable generator.

This is an example of how the APIs in this RFC allow for self-referential data
types to be created safely.

# Drawbacks
[drawbacks]: #drawbacks

This adds additional APIs to std, including an auto trait. Such additions
should not be taken lightly, and only included if they are well-justified by
the abstractions they express.

# Rationale and alternatives
[alternatives]: #alternatives

## Comparison to `?Move`

One previous proposal was to add a built-in `Move` trait, similar to `Sized`. A
type that did not implement `Move` could not be moved after it had been
referenced.

This solution had some problems. First, the `?Move` bound ended up "infecting"
many different APIs where it wasn't relevant, and introduced a breaking change
in several cases where the API bound changed in a non-backwards compatible way.

In a certain sense, this proposal is a much more narrowly scoped version of
`?Move`. With `?Move`, *any* reference could act as the "Pin" reference does
here. However, because of this flexibility, the negative consequences of having
a type that can't be moved had a much broader impact.

Instead, we require APIs to opt into supporting immovability (a niche case) by
operating with the `Pin` type, avoiding "infecting" the basic reference type
with concerns around immovable types.

## Comparison to using `unsafe` APIs

Another alternative we've considered was to just have the APIs which require
immovability be `unsafe`. It would be up to the users of these APIs to review
and guarantee that they never moved the self-referential types. For example,
generator would look like this:

```rust
trait Generator {
    type Yield;
    type Return;

    unsafe fn resume(&mut self) -> CoResult<Self::Yield, Self::Return>;
}
```

This would require no extensions to the standard library, but would place the
burden on every user who wants to call resume to guarantee (at the risk of
memory insafety) that their types were not moved, or that they were moveable.
This seemed like a worse trade off than adding these APIs.

## Anchor as a wrapper type and `StableDeref`

In a previous iteration of this RFC, `Anchor` was a wrapper type that could
"anchor" any smart pointer, and there was a hierarchy of traits relating to the
stability of the referent of different pointer types.

The primary benefit of this approach was that it was partially integrated with
crates like owning-ref and rental, which also use a hierarchy of stability
traits. However, because of differences in the requirements, the traits used by
owning-ref et al ended up being a non-overlapping subset of the traits proposed
by this RFC from the traits used by the Anchor type. Merging these into a
single hierarchy provided relatively little benefit.

And the only types that implemented all of the necessary traits to be put into
an Anchor before were `Box<T>` and `Vec<T>`. Because you cannot get mutable
access to the smart pointer (unless the referent implements `Move`), an
`Anchor<Vec<T>>` was not really any different from an `Anchor<Box<[T]>>` in the
previous iteration of the RFC. For this reason, making `Anchor` always a box,
and just supporting `Anchor<[T]>`, reduced the API complexity without losing
any expressiveness.

## Stack pinning API (potential future extension)

This API supports "anchoring" `!Move` types in the heap. However, they can also
be safely held in place in the stack, allowing a safe API for creating a `Pin`
referencing a stack allocated `!Move` type.

This API is small, and does not become a part of anyone's public API. For that
reason, we'll start by allowing it to grow out of tree, in third party crates,
before including it in std.

## Making `Pin` a built-in type (potential future extension)

The `Pin` type could instead be a new kind of first-class reference - `&'a pin
T`. This would have some advantages - it would be trivial to project through
fields, for example, and "stack pinning" would not require an API, it would be
natural. However, it has the downside of adding a new reference type, a very
big language change.

For now, we're happy to stick with the `Pin` struct in std, and if this type is
ever added, turn the `Pin` type into an alias for the reference type.

# Unresolved questions
[unresolved]: #unresolved-questions

The names used in this RFC are all entirely up for debate. Some of the items
introduced (especially the `Move` trait) have evolved away from their original
design, making the names a bit of a misnomer (`Move` really means that its safe
to convert between `Pin<T>` and `&mut T`, for example). We want to make sure we
have adequate names before stabilizing these APIs.

[stable-deref]: https://crates.io/crates/stable_deref_trait
