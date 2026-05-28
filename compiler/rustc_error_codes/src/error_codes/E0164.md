Something which is neither a tuple struct nor a tuple variant was used as a
pattern.

Erroneous code example:

```compile_fail,E0164
enum A {
    B,
    C,
}

impl A {
    fn new() {}
}

fn bar(foo: A) {
    match foo {
        A::new() => (), // error!
        _ => {}
    }
}
```

This error means that an attempt was made to match something which is neither a
tuple struct nor a tuple variant. Only these two elements are allowed as a
pattern:

```
enum A {
    B,
    C,
}

impl A {
    fn new() {}
}

fn bar(foo: A) {
    match foo {
        A::B => (), // ok!
        _ => {}
    }
}
```
