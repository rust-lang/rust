% Casting Between Types

Rust, with its focus on safety, provides two different ways of casting
different types between each other. The first, `as`, is for safe casts.
In contrast, `transmute` allows for arbitrary casting, and is one of the
most dangerous features of Rust!

# `as`

The `as` keyword does safe casting:

```rust
let x: i32 = 5;

let y = x as i64;
```

There are three major categories of safe cast: explicit coercions, casts
between numeric types, and pointer casts.

Casting is not transitive: even if `e as U1 as U2` is a valid
expression, `e as U2` is not necessarily so (in fact it will only be valid if
`U1` coerces to `U2`).


## Explicit coercions

A cast `e as U` is valid if `e` has type `T` and `T` *coerces* to `U`.

For example:

```rust
let a = "hello";
let b = a as String
```

Coercions always occur implicitly so this form is only for clarity.

## Numeric casts

A cast `e as U` is also valid in any of the following cases:

 * `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
 * `e` is a C-like enum and `U` is an integer type; *enum-cast*
 * `e` has type `bool` or `char` and `U` is an integer; *prim-int-cast*
 * `e` has type `u8` and `U` is `char`; *u8-char-cast*
 
For example

```rust
let one = true as u8;
let at_sign = 64 as char;
```
 
## Pointer casts
 
Perhaps surprisingly, it is safe to cast pointers to and from integers, and
to cast between pointers to different types subject to some constraints. It
is only unsafe to dereference the pointer.

* `e` has type `*T`, `U` is a pointer to `*U_0`, and either `U_0: Sized` or
  unsize_kind(`T`) = unsize_kind(`U_0`); a *ptr-ptr-cast*
* `e` has type `*T` and `U` is a numeric type, while `T: Sized`; *ptr-addr-cast*
* `e` is an integer and `U` is `*U_0`, while `U_0: Sized`; *addr-ptr-cast*
* `e` has type `&[T; n]` and `U` is `*const T`; *array-ptr-cast*
* `e` is a function pointer type and `U` has type `*T`,
  while `T: Sized`; *fptr-ptr-cast*
* `e` is a function pointer type and `U` is an integer; *fptr-addr-cast*


# `transmute`

`as` only allows safe casting, and will for example reject an attempt to
cast four bytes into a `u32`:

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

This is a ‘non-scalar cast’ because we have multiple values here: the four
elements of the array. These kinds of casts are very dangerous, because they
make assumptions about the way that multiple underlying structures are
implemented. For this, we need something more dangerous.

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
