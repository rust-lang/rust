`derive` was applied on something which is not a struct, a union or an enum.

Erroneous code example:

```compile_fail,E0774
trait Foo {
    #[derive(Clone)] // error!
    type Bar;
}
```

As said above, the `derive` attribute is only allowed on structs, unions or
enums:

```
#[derive(Clone)] // ok!
struct Bar {
    field: u32,
}
```

You can find more information about `derive` in the [Rust Book].

[Rust Book]: https://doc.rust-lang.org/book/appendix-03-derivable-traits.html
