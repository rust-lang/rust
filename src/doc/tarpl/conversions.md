% Type Conversions

At the end of the day, everything is just a pile of bits somewhere, and type
systems are just there to help us use those bits right. Needing to reinterpret
those piles of bits as different types is a common problem and Rust consequently
gives you several ways to do that.

First we'll look at the ways that *Safe Rust* gives you to reinterpret values.
The most trivial way to do this is to just destructure a value into its
constituent parts and then build a new type out of them. e.g.

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

But this is, at best, annoying to do. For common conversions, Rust provides
more ergonomic alternatives.

