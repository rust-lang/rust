% Exotically Sized Types

Most of the time, we think in terms of types with a fixed, positive size. This
is not always the case, however.





# Dynamically Sized Types (DSTs)

Rust also supports types without a statically known size. On the surface, this
is a bit nonsensical: Rust *must* know the size of something in order to work
with it! DSTs are generally produced as views, or through type-erasure of types
that *do* have a known size. Due to their lack of a statically known size, these
types can only exist *behind* some kind of pointer. They consequently produce a
*fat* pointer consisting of the pointer and the information that *completes*
them.

For instance, the slice type, `[T]`, is some statically unknown number of
elements stored contiguously. `&[T]` consequently consists of a `(&T, usize)`
pair that specifies where the slice starts, and how many elements it contains.
Similarly, Trait Objects support interface-oriented type erasure through a
`(data_ptr, vtable_ptr)` pair.

Structs can actually store a single DST directly as their last field, but this
makes them a DST as well:

```rust
// Can't be stored on the stack directly
struct Foo {
    info: u32,
    data: [u8],
}
```

**NOTE: As of Rust 1.0 struct DSTs are broken if the last field has
a variable position based on its alignment.**





# Zero Sized Types (ZSTs)

Rust actually allows types to be specified that occupy *no* space:

```rust
struct Foo; // No fields = no size

// All fields have no size = no size
struct Baz {
    foo: Foo,
    qux: (),      // empty tuple has no size
    baz: [u8; 0], // empty array has no size
}
```

On their own, ZSTs are, for obvious reasons, pretty useless. However as with
many curious layout choices in Rust, their potential is realized in a generic
context.

Rust largely understands that any operation that produces or stores a ZST can be
reduced to a no-op. For instance, a `HashSet<T>` can be effeciently implemented
as a thin wrapper around `HashMap<T, ()>` because all the operations `HashMap`
normally does to store and retrieve values will be completely stripped in
monomorphization.

Similarly `Result<(), ()>` and `Option<()>` are effectively just fancy `bool`s.

Safe code need not worry about ZSTs, but *unsafe* code must be careful about the
consequence of types with no size. In particular, pointer offsets are no-ops,
and standard allocators (including jemalloc, the one used by Rust) generally
consider passing in `0` as Undefined Behaviour.





# Empty Types

Rust also enables types to be declared that *cannot even be instantiated*. These
types can only be talked about at the type level, and never at the value level.

```rust
enum Foo { } // No variants = EMPTY
```

TODO: WHY?!
