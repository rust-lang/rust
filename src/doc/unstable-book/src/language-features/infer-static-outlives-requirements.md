# `infer_static_outlives_requirements`

The tracking issue for this feature is: [#44493]

[#44493]: https://github.com/rust-lang/rust/issues/44493

------------------------
The `infer_static_outlives_requirements` feature indicates that certain
`'static` outlives requirements can be inferred by the compiler rather than
stating them explicitly.

Note: It is an accompanying feature to `infer_outlives_requirements`,
which must be enabled to infer outlives requirements.

For example, currently generic struct definitions that contain
references, require where-clauses of the form T: 'static. By using
this feature the outlives predicates will be inferred, although
they may still be written explicitly.

```rust,ignore (pseudo-Rust)
struct Foo<U> where U: 'static { // <-- currently required
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}
```


## Examples:

```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]
#![feature(infer_static_outlives_requirements)]

#[rustc_outlives]
// Implicitly infer U: 'static
struct Foo<U> {
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}
```

