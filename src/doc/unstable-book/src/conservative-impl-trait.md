# `conservative_impl_trait`

The tracking issue for this feature is: [#34511]

[#34511]: https://github.com/rust-lang/rust/issues/34511

------------------------

The `conservative_impl_trait` feature allows a conservative form of abstract
return types.

Abstract return types allow a function to hide a concrete return type behind a
trait interface similar to trait objects, while still generating the same
statically dispatched code as with concrete types.

## Examples

```rust
#![feature(conservative_impl_trait)]

fn even_iter() -> impl Iterator<Item=u32> {
    (0..).map(|n| n * 2)
}

fn main() {
    let first_four_even_numbers = even_iter().take(4).collect::<Vec<_>>();
    assert_eq!(first_four_even_numbers, vec![0, 2, 4, 6]);
}
```

## Background

In today's Rust, you can write function signatures like:

````rust,ignore
fn consume_iter_static<I: Iterator<Item=u8>>(iter: I) { }

fn consume_iter_dynamic(iter: Box<Iterator<Item=u8>>) { }
````

In both cases, the function does not depend on the exact type of the argument.
The type held is "abstract", and is assumed only to satisfy a trait bound.

* In the `_static` version using generics, each use of the function is
  specialized to a concrete, statically-known type, giving static dispatch,
  inline layout, and other performance wins.
* In the `_dynamic` version using trait objects, the concrete argument type is
  only known at runtime using a vtable.

On the other hand, while you can write:

````rust,ignore
fn produce_iter_dynamic() -> Box<Iterator<Item=u8>> { }
````

...but you _cannot_ write something like:

````rust,ignore
fn produce_iter_static() -> Iterator<Item=u8> { }
````

That is, in today's Rust, abstract return types can only be written using trait
objects, which can be a significant performance penalty. This RFC proposes
"unboxed abstract types" as a way of achieving signatures like
`produce_iter_static`. Like generics, unboxed abstract types guarantee static
dispatch and inline data layout.
