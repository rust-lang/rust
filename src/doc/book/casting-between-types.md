% Casting Between Types

Rust, with its focus on safety, provides two different ways of casting
different types between each other. The first, `as`, is for safe casts.
In contrast, `transmute` allows for arbitrary casting, and is one of the
most dangerous features of Rust!

# Coercion

Coercion between types is implicit and has no syntax of its own, but can
be spelled out with [`as`](#explicit-coercions).

Coercion occurs in `let`, `const`, and `static` statements; in
function call arguments; in field values in struct initialization; and in a
function result.

The most common case of coercion is removing mutability from a reference:

 * `&mut T` to `&T`
 
An analogous conversion is to remove mutability from a
[raw pointer](raw-pointers.md):

 * `*mut T` to `*const T`
 
References can also be coerced to raw pointers:

 * `&T` to `*const T`

 * `&mut T` to `*mut T`

Custom coercion may be defined using [`Deref`](deref-coercions.md).

Coercion is transitive.
 
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
let b = a as String;
```

## Numeric casts

A cast `e as U` is also valid in any of the following cases:

 * `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
 * `e` is a C-like enum (with no data attached to the variants),
    and `U` is an integer type; *enum-cast*
 * `e` has type `bool` or `char` and `U` is an integer type; *prim-int-cast*
 * `e` has type `u8` and `U` is `char`; *u8-char-cast*
 
For example

```rust
let one = true as u8;
let at_sign = 64 as char;
let two_hundred = -56i8 as u8;
```

The semantics of numeric casts are:

* Casting between two integers of the same size (e.g. i32 -> u32) is a no-op
* Casting from a larger integer to a smaller integer (e.g. u32 -> u8) will
  truncate
* Casting from a smaller integer to a larger integer (e.g. u8 -> u32) will
    * zero-extend if the source is unsigned
    * sign-extend if the source is signed
* Casting from a float to an integer will round the float towards zero
    * **[NOTE: currently this will cause Undefined Behavior if the rounded
      value cannot be represented by the target integer type][float-int]**.
      This includes Inf and NaN. This is a bug and will be fixed.
* Casting from an integer to float will produce the floating point
  representation of the integer, rounded if necessary (rounding strategy
  unspecified)
* Casting from an f32 to an f64 is perfect and lossless
* Casting from an f64 to an f32 will produce the closest possible value
  (rounding strategy unspecified)
    * **[NOTE: currently this will cause Undefined Behavior if the value
      is finite but larger or smaller than the largest or smallest finite
      value representable by f32][float-float]**. This is a bug and will
      be fixed.

[float-int]: https://github.com/rust-lang/rust/issues/10184
[float-float]: https://github.com/rust-lang/rust/issues/15536
 
## Pointer casts
 
Perhaps surprisingly, it is safe to cast [raw pointers](raw-pointers.md) to and
from integers, and to cast between pointers to different types subject to
some constraints. It is only unsafe to dereference the pointer:

```rust
let a = 300 as *const char; // a pointer to location 300
let b = a as u32;
```

`e as U` is a valid pointer cast in any of the following cases:

* `e` has type `*T`, `U` has type `*U_0`, and either `U_0: Sized` or
  `unsize_kind(T) == unsize_kind(U_0)`; a *ptr-ptr-cast*
  
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
