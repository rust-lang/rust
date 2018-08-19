# `infer_outlives_requirements`

The tracking issue for this feature is: [#44493]

[#44493]: https://github.com/rust-lang/rust/issues/44493

------------------------
The `infer_outlives_requirements` feature indicates that certain
outlives requirements can be infered by the compiler rather than
stating them explicitly.

For example, currently generic struct definitions that contain
references, require where-clauses of the form T: 'a. By using
this feature the outlives predicates will be infered, although
they may still be written explicitly.

```rust,ignore (pseudo-Rust)
struct Foo<'a, T>
  where T: 'a // <-- currently required
  {
      bar: &'a T,
  }
```


## Examples:


```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]

// Implicitly infer T: 'a
struct Foo<'a, T> {
    bar: &'a T,
}
```

```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]

// Implicitly infer `U: 'b`
struct Foo<'b, U> {
    bar: Bar<'b, U>
}

struct Bar<'a, T> where T: 'a {
    x: &'a (),
    y: T,
}
```

```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]

// Implicitly infer `b': 'a`
struct Foo<'a, 'b, T> {
    x: &'a &'b T
}
```

```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]

// Implicitly infer `<T as std::iter::Iterator>::Item : 'a`
struct Foo<'a, T: Iterator> {
    bar: &'a T::Item
```
