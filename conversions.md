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




# Auto-Deref

(Maybe nix this in favour of receiver coercions)

Deref is a trait that allows you to overload the unary `*` to specify a type
you dereference to. This is largely only intended to be implemented by pointer
types like `&`, `Box`, and `Rc`. The dot operator will automatically perform
automatic dereferencing, so that foo.bar() will work uniformly on `Foo`, `&Foo`, `
&&Foo`, `&Rc<Box<&mut&Box<Foo>>>` and so-on. Search bottoms out on the *first* match,
so implementing methods on pointers is generally to be avoided, as it will shadow
"actual" methods.




# Coercions

Types can implicitly be coerced to change in certain contexts. These changes are
generally just *weakening* of types, largely focused around pointers and lifetimes.
They mostly exist to make Rust "just work" in more cases, and are largely harmless.

Here's all the kinds of coercion:


Coercion is allowed between the following types:

* `T` to `U` if `T` is a [subtype](lifetimes.html#subtyping-and-variance)
  of `U` (the 'identity' case);

* `T_1` to `T_3` where `T_1` coerces to `T_2` and `T_2` coerces to `T_3`
  (transitivity case);

* `&mut T` to `&T`;

* `*mut T` to `*const T`;

* `&T` to `*const T`;

* `&mut T` to `*mut T`;

* `T` to `U` if `T` implements `CoerceUnsized<U>` (see below) and `T = Foo<...>`
  and `U = Foo<...>`;

* From TyCtor(`T`) to TyCtor(coerce_inner(`T`));

where TyCtor(`T`) is one of `&T`, `&mut T`, `*const T`, `*mut T`, or `Box<T>`.
And where coerce_inner is defined as

* coerce_inner(`[T, ..n]`) = `[T]`;

* coerce_inner(`T`) = `U` where `T` is a concrete type which implements the
  trait `U`;

* coerce_inner(`T`) = `U` where `T` is a sub-trait of `U`;

* coerce_inner(`Foo<..., T, ...>`) = `Foo<..., coerce_inner(T), ...>` where
  `Foo` is a struct and only the last field has type `T` and `T` is not part of
  the type of any other fields;

* coerce_inner(`(..., T)`) = `(..., coerce_inner(T))`.

Coercions only occur at a *coercion site*. Exhaustively, the coercion sites
are:

* In `let` statements where an explicit type is given: in `let _: U = e;`, `e`
  is coerced to to have type `U`;

* In statics and consts, similarly to `let` statements;

* In argument position for function calls. The value being coerced is the actual
  parameter and it is coerced to the type of the formal parameter. For example,
  where `foo` is defined as `fn foo(x: U) { ... }` and is called with `foo(e);`,
  `e` is coerced to have type `U`;

* Where a field of a struct or variant is instantiated. E.g., where `struct Foo
  { x: U }` and the instantiation is `Foo { x: e }`, `e` is coerced to to have
  type `U`;

* The result of a function, either the final line of a block if it is not semi-
  colon terminated or any expression in a `return` statement. For example, for
  `fn foo() -> U { e }`, `e` is coerced to to have type `U`;

If the expression in one of these coercion sites is a coercion-propagating
expression, then the relevant sub-expressions in that expression are also
coercion sites. Propagation recurses from these new coercion sites. Propagating
expressions and their relevant sub-expressions are:

* array literals, where the array has type `[U, ..n]`, each sub-expression in
  the array literal is a coercion site for coercion to type `U`;

* array literals with repeating syntax, where the array has type `[U, ..n]`, the
  repeated sub-expression is a coercion site for coercion to type `U`;

* tuples, where a tuple is a coercion site to type `(U_0, U_1, ..., U_n)`, each
  sub-expression is a coercion site for the respective type, e.g., the zero-th
  sub-expression is a coercion site to `U_0`;

* the box expression, if the expression has type `Box<U>`, the sub-expression is
  a coercion site to `U`;

* parenthesised sub-expressions (`(e)`), if the expression has type `U`, then
  the sub-expression is a coercion site to `U`;

* blocks, if a block has type `U`, then the last expression in the block (if it
  is not semicolon-terminated) is a coercion site to `U`. This includes blocks
  which are part of control flow statements, such as `if`/`else`, if the block
  has a known type.


Note that we do not perform coercions when matching traits (except for
receivers, see below). If there is an impl for some type `U` and `T` coerces to
`U`, that does not constitute an implementation for `T`. For example, the
following will not type check, even though it is OK to coerce `t` to `&T` and
there is an impl for `&T`:

```
struct T;
trait Trait {}

fn foo<X: Trait>(t: X) {}

impl<'a> Trait for &'a T {}


fn main() {
    let t: &mut T = &mut T;
    foo(t); //~ ERROR failed to find an implementation of trait Trait for &mut T
}
```

In a cast expression, `e as U`, the compiler will first attempt to coerce `e` to
`U`, only if that fails will the conversion rules for casts (see below) be
applied.



TODO: receiver coercions?


# Casts

Casts are a superset of coercions: every coercion can be explicitly invoked via a
cast, but some conversions *require* a cast. These "true casts" are generally regarded
as dangerous or problematic actions. True casts revolve around raw pointers and
the primitive numeric types. True casts aren't checked.

Here's an exhaustive list of all the true casts:

 * `e` has type `T` and `T` coerces to `U`; *coercion-cast*
 * `e` has type `*T`, `U` is `*U_0`, and either `U_0: Sized` or
    unsize_kind(`T`) = unsize_kind(`U_0`); *ptr-ptr-cast*
 * `e` has type `*T` and `U` is a numeric type, while `T: Sized`; *ptr-addr-cast*
 * `e` is an integer and `U` is `*U_0`, while `U_0: Sized`; *addr-ptr-cast*
 * `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
 * `e` is a C-like enum and `U` is an integer type; *enum-cast*
 * `e` has type `bool` or `char` and `U` is an integer; *prim-int-cast*
 * `e` has type `u8` and `U` is `char`; *u8-char-cast*
 * `e` has type `&[T; n]` and `U` is `*const T`; *array-ptr-cast*
 * `e` is a function pointer type and `U` has type `*T`,
   while `T: Sized`; *fptr-ptr-cast*
 * `e` is a function pointer type and `U` is an integer; *fptr-addr-cast*

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
