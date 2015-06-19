% Type Conversions

At the end of the day, everything is just a pile of bits somewhere, and type systems
are just there to help us use those bits right. Needing to reinterpret those piles
of bits as different types is a common problem and Rust consequently gives you
several ways to do that.

# Safe Rust

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

## Auto-Deref

Deref is a trait that allows you to overload the unary `*` to specify a type
you dereference to. This is largely only intended to be implemented by pointer
types like `&`, `Box`, and `Rc`. The dot operator will automatically perform
automatic dereferencing, so that foo.bar() will work uniformly on `Foo`, `&Foo`, `&&Foo`,
`&Rc<Box<&mut&Box<Foo>>>` and so-on. Search bottoms out on the *first* match,
so implementing methods on pointers is generally to be avoided, as it will shadow
"actual" methods.

## Coercions

Types can implicitly be coerced to change in certain contexts. These changes are generally
just *weakening* of types, largely focused around pointers. They mostly exist to make
Rust "just work" in more cases. For instance
`&mut T` coerces to `&T`, and `&T` coerces to `*const T`. The most useful coercion you will
actually think about it is probably the general *Deref Coercion*: `&T` coerces to `&U` when
`T: Deref<U>`. This enables us to pass an `&String` where an `&str` is expected, for instance.

## Casts

Casts are a superset of coercions: every coercion can be explicitly invoked via a cast,
but some changes require a cast. These "true casts" are generally regarded as dangerous or
problematic actions. True casts revolves around raw pointers and the primitive numeric
types. Here's an exhaustive list of all the true casts:

TODO: gank the RFC for sweet casts

For number -> number casts, there are quite a few cases to consider:

* casting between two integers of the same size (e.g. i32 -> u32) is a no-op
* casting from a smaller integer to a bigger integer (e.g. u32 -> u8) will truncate
* casting from a larger integer to a smaller integer (e.g. u8 -> u32) will
    * zero-extend if unsigned
    * sign-extend if signed
* casting from a float to an integer will round the float towards zero.
    * **NOTE: currently this will cause Undefined Behaviour if the rounded
      value cannot be represented by the target integer type**. This is a bug
      and will be fixed.
* casting from an integer to float will produce the floating point representation
  of the integer, rounded if necessary (rounding strategy unspecified).
* casting from an f32 to an f64 is perfect and lossless.
* casting from an f64 to an f32 will produce the closest possible value
  (rounding strategy unspecified).
    * **NOTE: currently this will cause Undefined Behaviour if the value
      is finite but larger or smaller than the largest or smallest finite
      value representable by f32**. This is a bug and will be fixed.

The casts involving rawptrs also allow us to completely bypass type-safety
by re-interpretting a pointer of T to a pointer of U for arbitrary types, as
well as interpret integers as addresses. However it is impossible to actually
*capitalize* on this violation in Safe Rust, because derefencing a raw ptr is
`unsafe`.


## Conversion Traits

For full formal specification of all the kinds of coercions and coercion sites, see:
https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md



* Coercions
* Casts
* Conversion Traits (Into/As/...)

# Unsafe Rust

* raw ptr casts
* mem::transmute
