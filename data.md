% Data Representation in Rust

Low-level programming cares a lot about data layout. It's a big deal. It also pervasively
influences the rest of the language, so we're going to start by digging into how data is
represented in Rust.

# The `rust` repr

Rust gives you the following ways to lay out composite data:

* structs (named product types)
* tuples (anonymous product types)
* arrays (homogeneous product types)
* enums (named sum types -- tagged unions)

For all these, individual fields are aligned to their preferred alignment.
For primitives this is equal to
their size. For instance, a u32 will be aligned to a multiple of 32 bits, and a u16 will
be aligned to a multiple of 16 bits. Composite structures will have their size rounded
up to be a multiple of the highest alignment required by their fields, and an alignment
requirement equal to the highest alignment required by their fields. So for instance,

```rust
struct A {
    a: u8,
    c: u64,
    b: u32,
}
```

will have a size that is a multiple of 64-bits, and 64-bit alignment.

There is *no indirection* for these types; all data is stored contiguously as you would
expect in C. However with the exception of arrays, the layout of data is not by
default specified in Rust. Given the two following struct definitions:

```rust
struct A {
    a: i32,
    b: u64,
}

struct B {
    x: i32,
    b: u64,
}
```

Rust *does* guarantee that two instances of A have their data laid out in exactly
the same way. However Rust *does not* guarantee that an instance of A has the same
field ordering or padding as an instance of B (in practice there's no *particular*
reason why they wouldn't, other than that its not currently guaranteed).

With A and B as written, this is basically nonsensical, but several other features
of Rust make it desirable for the language to play with data layout in complex ways.

For instance, consider this struct:

```rust
struct Foo<T, U> {
    count: u16,
    data1: T,
    data2: U,
}
```

Now consider the monomorphizations of `Foo<u32, u16>` and `Foo<u16, u32>`. If Rust lays out the
fields in the order specified, we expect it to *pad* the values in the struct to satisfy
their *alignment* requirements. So if Rust didn't reorder fields, we would expect Rust to
produce the following:

```rust
struct Foo<u16, u32> {
    count: u16,
    data1: u16,
    data2: u32,
}

struct Foo<u32, u16> {
    count: u16,
    _pad1: u16,
    data1: u32,
    data2: u16,
    _pad2: u16,
}
```

The former case quite simply wastes space. An optimal use of space therefore requires
different monomorphizations to *have different field orderings*.

**Note: this is a hypothetical optimization that is not yet implemented in Rust 1.0.0**

Enums make this consideration even more complicated. Naively, an enum such as:

```rust
enum Foo {
    A(u32),
    B(u64),
    C(u8),
}
```

would be laid out as:

```rust
struct FooRepr {
    data: u64, // this is *really* either a u64, u32, or u8 based on `tag`
    tag: u8, // 0 = A, 1 = B, 2 = C
}
```

And indeed this is approximately how it would be laid out in general
(modulo the size and position of `tag`). However there are several cases where
such a representation is ineffiecient. The classic case of this is Rust's
"null pointer optimization". Given a pointer that is known to not be null
(e.g. `&u32`), an enum can *store* a discriminant bit *inside* the pointer
by using null as a special value. The net result is that
`sizeof(Option<&T>) == sizeof<&T>`

There are many types in Rust that are, or contain, "not null" pointers such as `Box<T>`, `Vec<T>`,
`String`, `&T`, and `&mut T`. Similarly, one can imagine nested enums pooling their tags into
a single descriminant, as they are by definition known to have a limited range of valid values.
In principle enums can use fairly elaborate algorithms to cache bits throughout nested types
with special constrained representations. As such it is *especially* desirable that we leave
enum layout unspecified today.

# Dynamically Sized Types (DSTs)

Rust also supports types without a statically known size. On the surface,
this is a bit nonsensical: Rust must know the size of something in order to
work with it. DSTs are generally produced as views, or through type-erasure
of types that *do* have a known size. Due to their lack of a statically known
size, these types can only exist *behind* some kind of pointer. They consequently
produce a *fat* pointer consisting of the pointer and the information that
*completes* them.

For instance, the slice type, `[T]`, is some statically unknown number of elements
stored contiguously. `&[T]` consequently consists of a `(&T, usize)` pair that specifies
where the slice starts, and how many elements it contains. Similarly Trait Objects
support interface-oriented type erasure through a `(data_ptr, vtable_ptr)` pair.

Structs can actually store a single DST directly as their last field, but this
makes them a DST as well:

```rust
// Can't be stored on the stack directly
struct Foo {
    info: u32,
    data: [u8],
}
```

# Zero Sized Types (ZSTs)

Rust actually allows types to be specified that occupy *no* space:

```rust
struct Foo; // No fields = no size
enum Bar; // No variants = no size

// All fields have no size = no size
struct Baz {
    foo: Foo,
    bar: Bar,
    qux: (), // empty tuple has no size
}
```

On their own, ZSTs are, for obvious reasons, pretty useless. However
as with many curious layout choices in Rust, their potential is realized in a generic
context.

Rust largely understands that any operation that produces or stores a ZST
can be reduced to a no-op. For instance, a `HashSet<T>` can be effeciently implemented
as a thin wrapper around `HashMap<T, ()>` because all the operations `HashMap` normally
does to store and retrieve keys will be completely stripped in monomorphization.

Similarly `Result<(), ()>` and `Option<()>` are effectively just fancy `bool`s.

Safe code need not worry about ZSTs, but *unsafe* code must be careful about the
consequence of types with no size. In particular, pointer offsets are no-ops, and
standard allocators (including jemalloc, the one used by Rust) generally consider
passing in `0` as Undefined Behaviour.

# Drop Flags

For unfortunate legacy implementation reasons, Rust as of 1.0.0 will do a nasty trick to
any type that implements the `Drop` trait (has a destructor): it will insert a secret field
in the type. That is,

```rust
struct Foo {
    a: u32,
    b: u32,
}

impl Drop for Foo {
    fn drop(&mut self) { }
}
```

will cause Foo to secretly become:

```rust
struct Foo {
    a: u32,
    b: u32,
    _drop_flag: u8,
}
```

For details as to *why* this is done, and how to make it not happen, check out
[SOME OTHER SECTION].

# Alternative representations

Rust allows you to specify alternative data layout strategies from the default Rust
one.

# repr(C)

This is the most important `repr`. It has fairly simple intent: do what C does.
The order, size, and alignment of fields is exactly what you would expect from
C or C++. Any type you expect to pass through an FFI boundary should have `repr(C)`,
as C is the lingua-franca of the programming world. However this is also necessary
to soundly do more elaborate tricks with data layout such as reintepretting values
as a different type.

However, the interaction with Rust's more exotic data layout features must be kept
in mind. Due to its dual purpose as a "for FFI" and "for layout control", repr(C)
can be applied to types that will be nonsensical or problematic if passed through
the FFI boundary.

* ZSTs are still zero-sized, even though this is not a standard behaviour
in C, and is explicitly contrary to the behaviour of an empty type in C++, which
still consumes a byte of space.

* DSTs are not a concept in C

* **The drop flag will still be added**

* This is equivalent to repr(u32) for enums (see below)

# repr(packed)

`repr(packed)` forces rust to strip any padding it would normally apply.
This may improve the memory footprint of a type, but will have negative
side-effects from "field access is heavily penalized" to "completely breaks
everything" based on target platform.

# repr(u8), repr(u16), repr(u32), repr(u64)

These specify the size to make a c-like enum (one which has no values in its variants).

