An identifier that is neither defined nor a struct was used.

Erroneous code example:

```compile_fail,E0422
fn main () {
    let x = Foo { x: 1, y: 2 };
}
```

In this case, `Foo` is undefined, so it inherently isn't anything, and
definitely not a struct.

```compile_fail
fn main () {
    let foo = 1;
    let x = foo { x: 1, y: 2 };
}
```

In this case, `foo` is defined, but is not a struct, so Rust can't use it as
one.
