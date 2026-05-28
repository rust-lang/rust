A literal value was used inside `#[derive]`.

Erroneous code example:

```compile_fail,E0777
#[derive("Clone")] // error!
struct Foo;
```

Only paths to traits are allowed as argument inside `#[derive]`. You can find
more information about the `#[derive]` attribute in the [Rust Book].


```
#[derive(Clone)] // ok!
struct Foo;
```

[Rust Book]: https://doc.rust-lang.org/book/appendix-03-derivable-traits.html
