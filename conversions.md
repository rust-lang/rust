% Type Conversions

At the end of the day, everything is just a pile of bits somewhere, and type systems
are just there to help us use those bits right. Needing to reinterpret those piles
of bits as different types is a common problem and Rust consequently gives you
several ways to do that.

First we'll look at the ways that *Safe Rust* gives you to reinterpret values. The
most trivial way to do this is to just destructure a value into its constituent
parts and then build a new type out of them. e.g.

```rust
struct Foo {
    x: u32,
    y: u16,
}

struct Bar {
    a: u32,
    b: u16,
}

fn reinterpret(foo: Foo) -> Bar {
    let Foo { x, y } = foo;
    Bar { a: x, b: y }
}
```

But this is, at best, annoying to do. For common conversions, rust provides
more ergonomic alternatives.




# Coercions

Types can implicitly be coerced to change in certain contexts. These changes are
generally just *weakening* of types, largely focused around pointers and lifetimes.
They mostly exist to make Rust "just work" in more cases, and are largely harmless.

Here's all the kinds of coercion:


Coercion is allowed between the following types:

* Subtyping: `T` to `U` if `T` is a [subtype](lifetimes.html#subtyping-and-variance)
  of `U`
* Transitivity: `T_1` to `T_3` where `T_1` coerces to `T_2` and `T_2` coerces to `T_3`
* Pointer Weakening:
    * `&mut T` to `&T`
    * `*mut T` to `*const T`
    * `&T` to `*const T`
    * `&mut T` to `*mut T`
* Unsizing: `T` to `U` if `T` implements `CoerceUnsized<U>`

`CoerceUnsized<Pointer<U>> for Pointer<T> where T: Unsize<U>` is implemented
for all pointer types (including smart pointers like Box and Rc). Unsize is
only implemented automatically, and enables the following transformations:

* `[T, ..n]` => `[T]`
* `T` => `Trait` where `T: Trait`
* `SubTrait` => `Trait` where `SubTrait: Trait` (TODO: is this now implied by the previous?)
* `Foo<..., T, ...>` => `Foo<..., U, ...>` where:
    * T: Unsize<U>
    * `Foo` is a struct
    * Only the last field has type `T`
    * `T` is not part of the type of any other fields

Coercions occur at a *coercion site*. Any location that is explicitly typed
will cause a coercion to its type. If inference is necessary, the coercion will
not be performed. Exhaustively, the coercion sites for an expression `e` to
type `U` are:

* let statements, statics, and consts: `let x: U = e`
* Arguments to functions: `takes_a_U(e)`
* Any expression that will be returned: `fn foo() -> U { e }`
* Struct literals: `Foo { some_u: e }`
* Array literals: `let x: [U; 10] = [e, ..]`
* Tuple literals: `let x: (U, ..) = (e, ..)`
* The last expression in a block: `let x: U = { ..; e }`

Note that we do not perform coercions when matching traits (except for
receivers, see below). If there is an impl for some type `U` and `T` coerces to
`U`, that does not constitute an implementation for `T`. For example, the
following will not type check, even though it is OK to coerce `t` to `&T` and
there is an impl for `&T`:

```rust
trait Trait {}

fn foo<X: Trait>(t: X) {}

impl<'a> Trait for &'a i32 {}


fn main() {
    let t: &mut i32 = &mut 0;
    foo(t);
}
```

```text
<anon>:10:5: 10:8 error: the trait `Trait` is not implemented for the type `&mut i32` [E0277]
<anon>:10     foo(t);
              ^~~
```




# The Dot Operator

The dot operator will perform a lot of magic to convert types. It will perform
auto-referencing, auto-dereferencing, and coercion until types match.

TODO: steal information from http://stackoverflow.com/questions/28519997/what-are-rusts-exact-auto-dereferencing-rules/28552082#28552082




# Casts

Casts are a superset of coercions: every coercion can be explicitly invoked via a
cast, but some conversions *require* a cast. These "true casts" are generally regarded
as dangerous or problematic actions. True casts revolve around raw pointers and
the primitive numeric types. True casts aren't checked.

Here's an exhaustive list of all the true casts. For brevity, we will use `*`
to denote either a `*const` or `*mut`, and `integer` to denote any integral primitive:

 * `*T as *U` where `T, U: Sized`
 * `*T as *U` TODO: explain unsized situation
 * `*T as integer`
 * `integer as *T`
 * `number as number`
 * `C-like-enum as integer`
 * `bool as integer`
 * `char as integer`
 * `u8 as char`
 * `&[T; n] as *const T`
 * `fn as *T` where `T: Sized`
 * `fn as integer`

where `&.T` and `*T` are references of either mutability,
and where unsize_kind(`T`) is the kind of the unsize info
in `T` - the vtable for a trait definition (e.g. `fmt::Display` or
`Iterator`, not `Iterator<Item=u8>`) or a length (or `()` if `T: Sized`).

Note that lengths are not adjusted when casting raw slices -
`T: *const [u16] as *const [u8]` creates a slice that only includes
half of the original memory.

Casting is not transitive, that is, even if `e as U1 as U2` is a valid
expression, `e as U2` is not necessarily so (in fact it will only be valid if
`U1` coerces to `U2`).

For numeric casts, there are quite a few cases to consider:

* casting between two integers of the same size (e.g. i32 -> u32) is a no-op
* casting from a larger integer to a smaller integer (e.g. u32 -> u8) will truncate
* casting from a smaller integer to a larger integer (e.g. u8 -> u32) will
    * zero-extend if the source is unsigned
    * sign-extend if the source is signed
* casting from a float to an integer will round the float towards zero
    * **NOTE: currently this will cause Undefined Behaviour if the rounded
      value cannot be represented by the target integer type**. This is a bug
      and will be fixed. (TODO: figure out what Inf and NaN do)
* casting from an integer to float will produce the floating point representation
  of the integer, rounded if necessary (rounding strategy unspecified).
* casting from an f32 to an f64 is perfect and lossless.
* casting from an f64 to an f32 will produce the closest possible value
  (rounding strategy unspecified).
    * **NOTE: currently this will cause Undefined Behaviour if the value
      is finite but larger or smaller than the largest or smallest finite
      value representable by f32**. This is a bug and will be fixed.





# Conversion Traits

TODO?




# Transmuting Types

Get out of our way type system! We're going to reinterpret these bits or die
trying! Even though this book is all about doing things that are unsafe, I really
can't emphasize that you should deeply think about finding Another Way than the
operations covered in this section. This is really, truly, the most horribly
unsafe thing you can do in Rust. The railguards here are dental floss.

`mem::transmute<T, U>` takes a value of type `T` and reinterprets it to have
type `U`. The only restriction is that the `T` and `U` are verified to have the
same size. The ways to cause Undefined Behaviour with this are mind boggling.

* First and foremost, creating an instance of *any* type with an invalid state
  is going to cause arbitrary chaos that can't really be predicted.
* Transmute has an overloaded return type. If you do not specify the return type
  it may produce a surprising type to satisfy inference.
* Making a primitive with an invalid value is UB
* Transmuting between non-repr(C) types is UB
* Transmuting an & to &mut is UB
* Transmuting to a reference without an explicitly provided lifetime
  produces an [unbound lifetime](lifetimes.html#unbounded-lifetimes)

`mem::transmute_copy<T, U>` somehow manages to be *even more* wildly unsafe than
this. It copies `size_of<U>` bytes out of an `&T` and interprets them as a `U`.
The size check that `mem::transmute` has is gone (as it may be valid to copy
out a prefix), though it is Undefined Behaviour for `U` to be larger than `T`.

Also of course you can get most of the functionality of these functions using
pointer casts.
