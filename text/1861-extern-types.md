- Feature Name: extern_types
- Start Date: 2017-01-18
- RFC PR: https://github.com/rust-lang/rfcs/pull/1861
- Rust Issue: https://github.com/rust-lang/rust/issues/43467

# Summary
[summary]: #summary

Add an `extern type` syntax for declaring types which are opaque to Rust's type
system.

# Motivation
[motivation]: #motivation

When interacting with external libraries we often need to be able to handle pointers to data that we don't know the size or layout of.

In C it's possible to declare a type but not define it.
These incomplete types can only be used behind pointers, a compilation error will result if the user tries to use them in such a way that the compiler would need to know their layout.

In Rust, we don't have this feature. Instead, a couple of problematic hacks are used in its place.

One is, we define the type as an uninhabited type. eg.

```rust
enum MyFfiType {}
```

Another is, we define the type with a private field and no methods to construct it.

```rust
struct MyFfiType {
    _priv: (),
}
```

The point of both these constructions is to prevent the user from being able to create or deal directly with instances of the type.
Neither of these types accurately reflect the reality of the situation.
The first definition is logically problematic as it defines a type which can never exist.
This means that references to the type can also—logically—never exist and raw pointers to the type are guaranteed to be
invalid.
The second definition says that the type is a ZST, that we can store it on the stack and that we can call `ptr::read`, `mem::size_of` etc. on it.
None of this is of course valid.

The controversies on how to represent foreign types even extend to the standard library too; see the discussion in the [libc_types RFC PR](https://github.com/rust-lang/rfcs/pull/1783).

This RFC instead proposes a way to directly express that a type exists but is unknown to Rust.

Finally, In the 2017 roadmap, [integration with other languages](https://github.com/rust-lang/rfcs/blob/master/text/1774-roadmap-2017.md#integration-with-other-languages), is listed as a priority.
Just like unions, this is an unsafe feature necessary for dealing with legacy code in a correct and understandable manner.

# Detailed design
[design]: #detailed-design

Add a new kind of type declaration, an extern type:

```rust
extern {
    type Foo;
}
```

These types are FFI-safe. They are also DSTs, meaning that they do not implement `Sized`. Being DSTs, they cannot be kept on the stack, can only be accessed through pointers and references and cannot be moved from.

In Rust, pointers to DSTs carry metadata about the object being pointed to.
For strings and slices this is the length of the buffer, for trait objects this is the object's vtable.
For extern types the metadata is simply `()`.
This means that a pointer to an extern type has the same size as a `usize` (ie. it is not a "fat pointer").
It also means that if we store an extern type at the end of a container (such as a struct or tuple) pointers to that container will also be identical to raw pointers (despite the container as a whole being unsized).
This is useful to support a pattern found in some C APIs where structs are passed around which have arbitrary data appended to the end of them: eg.

```rust
extern {
    type OpaqueTail;
}

#[repr(C)]
struct FfiStruct {
    data: u8,
    more_data: u32,
    tail: OpaqueTail,
}
```

As a DST, `size_of` and `align_of` do not work, but we must also be careful that `size_of_val` and `align_of_val` do not work either, as there is not necessarily a way at run-time to get the size of extern types either.
For an initial implementation, those methods can just panic, but before this is stabilized there should be some trait bound or similar on them that prevents their use statically.
The exact mechanism is more the domain of the custom DST RFC, [RFC 1524](https://github.com/rust-lang/rfcs/pull/1524), and so figuring that mechanism out will be delegated to it.

C's "pointer `void`" (not `()`, but the `void` used in `void*` and similar) is currently defined in two official places: [`std::os::raw::c_void`](https://doc.rust-lang.org/stable/std/os/raw/enum.c_void.html) and [`libc::c_void`](https://doc.rust-lang.org/libc/x86_64-unknown-linux-gnu/libc/enum.c_void.html).
Unifying these is out of scope for this RFC, but this feature should be used in their definition instead of the current tricks.
Strictly speaking, this is a breaking change, but the `std` docs explicitly say that `void` shouldn't be used without indirection.
And `libc` can, in the worst-case, make a breaking change.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Really, the question is "how do we teach *without* this".
As described above, the current tricks for doing this are wrong.
Furthermore, they are quite advanced touching upon many advanced corners of the language: zero-sized and uninhabited types are phenomena few programmer coming from mainstream languages have encountered.
From reading around other RFCs, issues, and internal threads, one gets a sense of two issues:
First, even among the group of Rust programmers enthusiastic enough to participate in these fora, the semantics of foreign types are not widely understood.
Second, there is annoyance that none of the current tricks, by nature of them all being flawed in different ways, would become standard.

By contrast, `extern type` does exactly what one wants, with an obvious and guessable syntax, without forcing the user to immediately understand all the nuance about why *these* semantics are indeed the right ones.
As they see various options fail: moves, stack variables, they can discover these semantics incrementally.
The benefits are such that this would soon displace the current hacks, making code in the wild more readable through consistent use of a pattern.

This should be taught in the foreign function interface chapter of the rust book in place of where it currently tells people to use uninhabited enums (ack!).

# Drawbacks
[drawbacks]: #drawbacks

Very slight addition of complexity to the language.

The syntax has the potential to be confused with introducing a type alias, rather than a new nominal type.
The use of `extern` here is also a bit of a misnomer as the name of the type does not refer to anything external to Rust.

# Alternatives
[alternatives]: #alternatives

Not do this.

Alternatively, rather than provide a way to create opaque types, we could just offer one distinguished type (`std::mem::OpaqueData` or something like that).
Then, to create new opaque types, users just declare a struct with a member of type `OpaqueData`.
This has the advantage of introducing no new syntax, and issues like FFI-compatibility would fall out of existing rules.

Another alternative is to drop the `extern` and allow a declaration to be written `type A;`.
This removes the (arguably disingenuous) use of the `extern` keyword although it makes the syntax look even more like a type alias.

# Unresolved questions
[unresolved]: #unresolved-questions

- Should we allow generic lifetime and type parameters on extern types?
  If so, how do they effect the type in terms of variance?

- [In std's source](https://github.com/rust-lang/rust/blob/164619a8cfe6d376d25bd3a6a9a5f2856c8de64d/src/libstd/os/raw.rs#L59-L64), it is mentioned that LLVM expects `i8*` for C's `void*`.
  We'd need to continue to hack this for the two `c_void`s in std and libc.
  But perhaps this should be done across-the-board for all extern types?
  Somebody should check what Clang does.
