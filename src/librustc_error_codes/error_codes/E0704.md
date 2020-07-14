An incorrect visibility restriction was specified.

Erroneous code example:

```compile_fail,E0704
mod foo {
    pub(foo) struct Bar {
        x: i32
    }
}
```

To make struct `Bar` only visible in module `foo` the `in` keyword should be
used:

```
mod foo {
    pub(in crate::foo) struct Bar {
        x: i32
    }
}
# fn main() {}
```

For more information see the Rust Reference on [Visibility].

[Visibility]: https://doc.rust-lang.org/reference/visibility-and-privacy.html
