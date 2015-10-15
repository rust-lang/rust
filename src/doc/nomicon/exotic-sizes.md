% Exotically Sized Types

Most of the time, we think in terms of types with a fixed, positive size. This
is not always the case, however.





# Dynamically Sized Types (DSTs)

Rust in fact supports Dynamically Sized Types (DSTs): types without a statically
known size or alignment. On the surface, this is a bit nonsensical: Rust *must*
know the size and alignment of something in order to correctly work with it! In
this regard, DSTs are not normal types. Due to their lack of a statically known
size, these types can only exist behind some kind of pointer. Any pointer to a
DST consequently becomes a *fat* pointer consisting of the pointer and the
information that "completes" them (more on this below).

There are two major DSTs exposed by the language: trait objects, and slices.

A trait object represents some type that implements the traits it specifies.
The exact original type is *erased* in favor of runtime reflection
with a vtable containing all the information necessary to use the type.
This is the information that completes a trait object: a pointer to its vtable.

A slice is simply a view into some contiguous storage -- typically an array or
`Vec`. The information that completes a slice is just the number of elements
it points to.

Structs can actually store a single DST directly as their last field, but this
makes them a DST as well:

```rust
// Can't be stored on the stack directly
struct Foo {
    info: u32,
    data: [u8],
}
```

**NOTE: [As of Rust 1.0 struct DSTs are broken if the last field has
a variable position based on its alignment][dst-issue].**





# Zero Sized Types (ZSTs)

Rust actually allows types to be specified that occupy no space:

```rust
struct Foo; // No fields = no size

// All fields have no size = no size
struct Baz {
    foo: Foo,
    qux: (),      // empty tuple has no size
    baz: [u8; 0], // empty array has no size
}
```

On their own, Zero Sized Types (ZSTs) are, for obvious reasons, pretty useless.
However as with many curious layout choices in Rust, their potential is realized
in a generic context: Rust largely understands that any operation that  produces
or stores a ZST can be reduced to a no-op. First off, storing it  doesn't even
make sense -- it doesn't occupy any space. Also there's only one  value of that
type, so anything that loads it can just produce it from the  aether -- which is
also a no-op since it doesn't occupy any space.

One of the most extreme example's of this is Sets and Maps. Given a
`Map<Key, Value>`, it is common to implement a `Set<Key>` as just a thin wrapper
around `Map<Key, UselessJunk>`. In many languages, this would necessitate
allocating space for UselessJunk and doing work to store and load UselessJunk
only to discard it. Proving this unnecessary would be a difficult analysis for
the compiler.

However in Rust, we can just say that  `Set<Key> = Map<Key, ()>`. Now Rust
statically knows that every load and store is useless, and no allocation has any
size. The result is that the monomorphized code is basically a custom
implementation of a HashSet with none of the overhead that HashMap would have to
support values.

Safe code need not worry about ZSTs, but *unsafe* code must be careful about the
consequence of types with no size. In particular, pointer offsets are no-ops,
and standard allocators (including jemalloc, the one used by default in Rust)
may return `nullptr` when a zero-sized allocation is requested, which is
indistinguishable from out of memory.





# Empty Types

Rust also enables types to be declared that *cannot even be instantiated*. These
types can only be talked about at the type level, and never at the value level.
Empty types can be declared by specifying an enum with no variants:

```rust
enum Void {} // No variants = EMPTY
```

Empty types are even more marginal than ZSTs. The primary motivating example for
Void types is type-level unreachability. For instance, suppose an API needs to
return a Result in general, but a specific case actually is infallible. It's
actually possible to communicate this at the type level by returning a
`Result<T, Void>`. Consumers of the API can confidently unwrap such a Result
knowing that it's *statically impossible* for this value to be an `Err`, as
this would require providing a value of type `Void`.

In principle, Rust can do some interesting analyses and optimizations based
on this fact. For instance, `Result<T, Void>` could be represented as just `T`,
because the `Err` case doesn't actually exist. The following *could* also
compile:

```rust,ignore
enum Void {}

let res: Result<u32, Void> = Ok(0);

// Err doesn't exist anymore, so Ok is actually irrefutable.
let Ok(num) = res;
```

But neither of these tricks work today, so all Void types get you is
the ability to be confident that certain situations are statically impossible.

One final subtle detail about empty types is that raw pointers to them are
actually valid to construct, but dereferencing them is Undefined Behavior
because that doesn't actually make sense. That is, you could model C's `void *`
type with `*const Void`, but this doesn't necessarily gain anything over using
e.g. `*const ()`, which *is* safe to randomly dereference.


[dst-issue]: https://github.com/rust-lang/rust/issues/26403
