- Feature Name: extern_types
- Start Date: 2017-01-18
- RFC PR: 
- Rust Issue: 

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

    enum MyFfiType {}

Another is, we define the type with a private field and no methods to construct it.

    struct MyFfiType {
        _priv: (),
    }

The point of both these constructions is to prevent the user from being able to create or deal directly with instances of the type.
Neither of these types accurately reflect the reality of the situation.
The first definition is logically problematic as it defines a type which can never exist.
This means that references to the type can also—logically—never exist and raw pointers to the type are guaranteed to be
invalid.
The second definition says that the type is a ZST, that we can store it on the stack and that we can call `ptr::read`, `mem::size_of` etc. on it.
None of this is of course valid.

This RFC instead proposes a way to directly express that a type exists but is unknown to Rust.

Finally, In the 2017 roadmap, [integration with other languages](https://github.com/rust-lang/rfcs/blob/master/text/1774-roadmap-2017.md#integration-with-other-languages), is listed as a priority.
Just like unions, this is an unsafe feature necessary for dealing with legacy code in a correct and understandable manner.

# Detailed design
[design]: #detailed-design

Add a new kind of type declaration, an `extern type`:

    extern type Foo;

These can also be declared inside an `extern` block:

    extern {
        type Foo;
    }

These types are FFI-safe. They are also DSTs, meaning that they implement `?Sized`. Being DSTs, they cannot be kept on the stack and can only be accessed through pointers.

In Rust, pointers to DSTs carry metadata about the object being pointed to.
For strings and slices this is the length of the buffer, for trait objects this is the object's vtable. For extern types the metadata is simply `()`.
This means that a pointer to an extern type is identical to a raw pointer.
It also means that if we store an extern type at the end of a container (such as a struct or tuple) pointers to that container will also be identical to raw pointers (despite the container as a whole being unsized).
This is useful to support a pattern found in some C APIs where structs are passed around which have arbitrary data appended to the end of them: eg.

```rust
extern type OpaqueTail;

#[repr(C)]
struct FfiStruct {
    data: u8,
    more_data: u32,
    tail: OpaqueTail,
}
```

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Really, the question is "how do we teach *without* this".
As described above, the current tricks for doing this are wrong.
Furthermore, they are quite advanced touching upon many advanced corners of the language: zero-sized and uninhabited types are phenomena few programmer coming from mainstream languages have considered.
From reading around other RFCs, issues, and internal threads, one gets a sense of two issues:
First, even among the group Rust programmers enthusiastic enough to participate in these fora, the semantics of foreign types are not widely understood.
Send, there is annoyance that none of the current tricks, by nature of them all being flawed in different ways, would become standard.

By contrast, `extern type` does exactly what one wants, with an obvious and guessable syntax, without forcing the user to immediately understand all the nuance about why *these* semantics are indeed the right ones.
As they see various options fail: moves, stack variables, they can discover these semantics incrementally.
The benefits are such that this would soon displace the current hacks, making code in the wild more readable through consistent use of a pattern.

This should be taught in the foreign function interface chapter of the rust book in place of where it currently tells people to use uninhabited enums (ack!).

# Drawbacks
[drawbacks]: #drawbacks

Very slight addition of complexity to the language.

# Alternatives
[alternatives]: #alternatives

Not do this.

# Unresolved questions
[unresolved]: #unresolved-questions

Should we allow generic lifetime and type parameters on extern types?
If so, how do they effect the type in terms of variance?
