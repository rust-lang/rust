% Casting Between Types

Rust, with its focus on safety, provides two different ways of casting
different types between each other. The first, `as`, is for safe casts.
In contrast, `transmute` allows for arbitrary casting, and is one of the
most dangerous features of Rust!

# `as`

The `as` keyword does basic casting:

```rust
let x: i32 = 5;

let y = x as i64;
```

It only allows certain kinds of casting, however:

```rust,ignore
let a = [0u8, 0u8, 0u8, 0u8];

let b = a as u32; // four eights makes 32
```

This errors with:

```text
error: non-scalar cast: `[u8; 4]` as `u32`
let b = a as u32; // four eights makes 32
        ^~~~~~~~
```

It’s a ‘non-scalar cast’ because we have multiple values here: the four
elements of the array. These kinds of casts are very dangerous, because they
make assumptions about the way that multiple underlying structures are
implemented. For this, we need something more dangerous.

# `transmute`

The `transmute` function is provided by a [compiler intrinsic][intrinsics], and
what it does is very simple, but very scary. It tells Rust to treat a value of
one type as though it were another type. It does this regardless of the
typechecking system, and just completely trusts you.

[intrinsics]: intrinsics.html

In our previous example, we know that an array of four `u8`s represents a `u32`
properly, and so we want to do the cast. Using `transmute` instead of `as`,
Rust lets us:

```rust
use std::mem;

unsafe {
    let a = [0u8, 0u8, 0u8, 0u8];

    let b = mem::transmute::<[u8; 4], u32>(a);
}
```

We have to wrap the operation in an `unsafe` block for this to compile
successfully. Technically, only the `mem::transmute` call itself needs to be in
the block, but it's nice in this case to enclose everything related, so you
know where to look. In this case, the details about `a` are also important, and
so they're in the block. You'll see code in either style, sometimes the context
is too far away, and wrapping all of the code in `unsafe` isn't a great idea.

While `transmute` does very little checking, it will at least make sure that
the types are the same size. This errors:

```rust,ignore
use std::mem;

unsafe {
    let a = [0u8, 0u8, 0u8, 0u8];

    let b = mem::transmute::<[u8; 4], u64>(a);
}
```

with:

```text
error: transmute called with differently sized types: [u8; 4] (32 bits) to u64
(64 bits)
```

Other than that, you're on your own!
