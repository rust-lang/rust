A struct's or struct-like enum variant's field was specified more than once.

Erroneous code example:

```compile_fail,E0062
struct Foo {
    x: i32,
}

fn main() {
    let x = Foo {
                x: 0,
                x: 0, // error: field `x` specified more than once
            };
}
```

This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was specified more than once. Each field should
be specified exactly one time. Example:

```
struct Foo {
    x: i32,
}

fn main() {
    let x = Foo { x: 0 }; // ok!
}
```
