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

# Explanation
[explanation]: #explanation

## The `Move` auto trait

This new auto trait is added to the `core::marker` and `std::marker` modules:

```rust
pub unsafe auto trait Move { }
```

A type implements `Move` if in its stack representation, it does not contain
internal references to other positions within its stack representation. Nearly
every type in Rust is `Move`.

Positive impls of `Move` are added for types which contain pointers to generic
types, but do not contain those types in their stack representation, e.g:

```rust
unsafe impl<'a, T: ?Sized> Move for &'a T { }
unsafe impl<'a, T: ?Sized> Move for &'a mut T { }
unsafe impl<'a, T: ?Sized> Move for Box<T> { }
unsafe impl<"a, T> Move for Vec<T> { }
// etc
```

This trait is a lang item, but only to generate negative impls for certain
generators. Unlike previous `?Move` proposals, and unlike some traits like
`Sized` and `Copy`, this trait does not impose any particular semantics on
types that do or don't implement it.

## The stability marker traits

A hierarchy of marker traits for smart pointer types is added to `core::marker`
and `std::marker`. These exist to provide a shared language for talking about
the guarantees that different smart pointers provide. This enables both the
kinds of self-referential support we talk about later in this RFC and other
APIs like rental and owning-ref.

### `Own` and `Share`

```rust
unsafe trait Own: Deref { }
unsafe trait Share: Deref + Clone { }
```

These two traits are for smart pointers which implement some form of ownership
construct.

- **Own** implies that this type has unique ownership over the data which it
  dereferences to. That is, unless the data is moved out of the smart pointer,
  when this pointer is destroyed, so too will that data. Examples of `Own`
  types are `Box<T>`, `Vec<T>` and `String`.
- **Share** implies that this type has shared ownership over the data which it
  dereferences to. It implies `Clone`, and every type it is `Clone`d it must
  continue to refer to the same data; it cannot perform deep clones of that
  data. Examples of `Share` types are `Rc<T>` and `Arc<T>`.

These traits are mutually exclusive - it would be a logic error to implement
both of them for a single type. We retain the liberty to assume that no type
ever does implement both - we could upgrade this from a logic error to
undefined behavior, we could make changes that would break any code that
implements both traits for the same type.

### `StableDeref` and `StableDerefMut`

```rust
unsafe trait StableDeref: Deref { }
unsafe trait StableDerefMut: StableDeref + DerefMut { }
```

These two traits are for any pointers which guarantee that the type they
dereference to is at a stable address. That is, moving the pointer does not
move the type being addressed.

- **StableDeref** implies that the referenced data will not move if you move
  this type or dereference it *immutably*. Types that implement this include
  `Box<T>`, both reference types, `Rc<T>`, `Arc<T>`, `Vec<T>`, and `String`.
  Pretty much everything in std that implements `Deref` implements
  `StableDeref`.
- **StableDerefMut** implies the same guarantees as `StableDeref`, but also
  guarantees that dereferencing a *mutable* reference will not cause the
  referenced data to change addresses. Because of this, it also implies
  `DerefMut`. Examples of type that implement this include `&mut T`, `Box<T>`,
  and `Vec<T>`.

Note that `StableDerefMut` does not imply that taking a mutable reference to
the smart pointer will not cause the referenced data to move. For example,
calling `push` on a `Vec` can cause the slice it dereferences to to change
locations. Its only obtaining a mutable reference to the target data which is
guaranteed not to relocate it.

Note also that this RFC does not propose implementing `StableDerefMut` for
`String`. This is to be forward compatible with the static string optimization,
an optimization which allows values of `&'static str` to be converted to
`String` without incurring a heap allocation. A component of this optimization
would cause `String` to allocate when dereferencing to an `&mut str` if the
backing data would otherwise be in rodata.

### Notes on existing ecosystem traits

These traits supplant certain traits in the ecosystem which already provide
similar guarantees. In particular, the [stable-deref][stable-deref] crate
provides a similar bit different hierarchy. The differences are:

- That crate draws no distinction between `StableDeref` and `StableDerefMut`.
  This does not leave forward compatibility for the static string optimization
  mentioned previously.
- That crate has no equivalent to the `Own` trait, which is necessary for some
  APIs using internal references.
- That crate has a `StableDerefClone` type, which is equivalent to the bound
  `Share + StableDeref` in our system.

If the hierarchy proposed in this RFC becomes stable, all users are encouraged
to migrate from that crate to the standard library traits.

## The `Pin` type

The `Pin` type is a wrapper around a mutable reference. If the type it
references is `!Move`, the `Pin` type guarantees that the referenced data will
never be moved again. It has a relatively small API. It is added to both
`std::mem` and `core::mem`.

```rust
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
unsafe (however, `Pin` can be easily immutably dereferenced, even for `!Move`
types).

The contract on the unsafe part of `Pin`s API is that a Pin cannot be
constructed if the data it references would ever move again, and that it cannot
be converted into a mutable reference if the data might ever be moved out of
that reference. In other words, if you have a `Pin` containing data which does
not implement `Move`, you have a guarantee that that data will never move.

The next two subsections describe safe APIs for constructing a `Pin` of data
which cannot be moved - one in the heap, and one in the stack.

### Pinning to the heap: The `Anchor` type

The `Anchor` wrapper takes a type that implements `StableDeref` and `Own`, and
prevents users from moving data out of that unless it implements `Move`. It is
added to `std::mem` and `core::mem`.

```rust
struct Anchor<T> {
     ptr: T,
}

impl<T: StableDerefMut + Own> Anchor<T> {
     pub fn new(ptr: T) -> Anchor<T>;

     pub unsafe fn get_mut(this: &mut Anchor<T>) -> &mut T;

     pub unsafe fn into_inner_unchecked(this: Anchor<T>) -> T;

     pub fn pin<'a>(this: &'a mut Anchor<T>) -> Pin<'a, T::Target>;
}

impl<T: StableDerefMut + Own> Anchor<T> where T::Target: Move {
     pub fn into_inner(this: Anchor<T>) -> T;
}

impl<T: StableDerefMut + Own> Deref for Anchor<T> {
     type Target = T;
}

impl<T: StableDerefMut + Own> DerefMut for Anchor<T> where T::Target: Move { }
```

Because `Anchor` implements `StableDeref` and `Own`, and it is not safe to get
an `&mut T` if the target of `T ` does not implement `Move`, an anchor
guarantees that the target of `T` will never move again. This satisfies the
safety constraints of `Pin`, allowing a user to construct a `Pin` from an
anchored pointer.

Because the data is anchored into the heap, you can move the anchor around
without moving the data itself. This makes anchor a very flexible way to handle
immovable data, at the cost of a heap allocation.

An example use:

```rust
let mut anchor = Anchor::new(Box::new(immovable_data));
let pin = Anchor::pin(&mut anchor);
```

### Pinning to the stack: `Pin::stack` and `pinned`

Data can also be pinned to the stack. This avoids the heap allocation, but the
pin must not outlive the data being pinned, and the API is less convenient.

First, the pinned function, added to `std::mem` and `core::mem`:

```rust
pub struct StackPinned<'a, T: ?Sized> {
     _marker: PhantomData<&'a mut &'a ()>,
     data: T
}

pub fn pinned<'a, T>(data: T) -> StackPinned<'a, T> {
     StackPinned {
          data, _marker: PhantomData,
     }
}
```

Second, this constructor is added to `Pin`:

```rust
impl<'a, T: ?Sized> Pin<'a, T> {
    pub fn stack(data: &'a mut StackPinned<'a, T>) -> Pin<'a, T>;
}
```

Because the lifetime of the `StackPinned` and the lifetime of the reference to
it are bound together, the StackPinned wrapper is functionally moved (and with
it the data inside it) into the Pin. Thus, even though the data is allocated on
the stack, it is pinned to its location for the remainder of its scope.

```rust
let mut data = mem::pinned(immovable_data);
let pin = Pin::stack(&mut data);
```

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

## Stabilization planning

This RFC proposes a large API addition, and so it is broken down into four
separate feature flags, which can be stabilized in stages:

1. `stable_deref` - to control the smart pointer trait hierarchy - StableDeref,
   StableDerefMut, Own, and Share.
2. `pin_and_move` - to control the `Move` auto trait and the `Pin` type. These
   two components only make sense working together.
3. `anchor` - to control the `Anchor` struct, pinning to the heap.
4. `stack_pinning` - to control the APIs related to stack pinning.

# Drawbacks
[drawbacks]: #drawbacks

This adds additional APIs to std, including several marker traits and an auto
trait. Such additions should not be taken lightly, and only included if they
are well-justified by the abstractions they express.

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

## Relationship to owning-ref & rental

Existing crates like owning-ref and rental make some use of "self-referential"
types. Unlike the generators this RFC is designed to support, their references
always point into the heap - making it acceptable to move their types around.

However, some of this infrastructure is still useful to those crates. In
particular, the stable deref hierarchy is related to the existing hierarchy in
the stable_deref crate, which those other crates depend on. By uplifting those
markers into the standard library, we create a shared, endorsed, and guaranteed
set of markers for the invariants those libraries care about.

In order to be implemented in safe code, those library need additional features
connecting to "existential" or "generative" lifetimes. These language changes
are out of scope for this RFC.

# Unresolved questions
[unresolved]: #unresolved-questions

The names used in this RFC are all entirely up for debate. Some of the items
introduced (especially the `Move` trait) have evolved away from their original
design, making the names a bit of a misnomer (`Move` really means that its safe
to convert between `Pin<T>` and `&mut T`, for example). We want to make sure we
have adequate names before stabilizing these APIs.
