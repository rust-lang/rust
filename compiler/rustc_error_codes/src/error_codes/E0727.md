A `yield` clause was used in an `async` context.

Erroneous code example:

```compile_fail,E0727,edition2018
#![feature(generators)]

fn main() {
    let generator = || {
        async {
            yield;
        }
    };
}
```

Here, the `yield` keyword is used in an `async` block,
which is not yet supported.

To fix this error, you have to move `yield` out of the `async` block:

```edition2018
#![feature(generators)]

fn main() {
    let generator = || {
        yield;
    };
}
```
