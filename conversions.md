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
problematic actions. The set of true casts is actually quite small, and once again revolves
largely around pointers. However it also introduces the primary mechanism to convert between
numeric types.

* rawptr -> rawptr (e.g. `*mut T as *const T` or `*mut T as *mut U`)
* rawptr <-> usize (e.g. `*mut T as usize` or `usize as *mut T`)
* primitive -> primitive (e.g. `u32 as u8` or `u8 as u32`)
* c-like enum -> integer/bool (e.g. `DaysOfWeek as u8`)
* `u8` -> `char`


## Conversion Traits

For full formal specification of all the kinds of coercions and coercion sites, see:
https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md



* Coercions
* Casts
* Conversion Traits (Into/As/...)

# Unsafe Rust

* raw ptr casts
* mem::transmute
