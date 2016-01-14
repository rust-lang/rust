% Using Rust Without the Standard Library

Rust’s standard library provides a lot of useful functionality, but assumes
support for various features of its host system: threads, networking, heap
allocation, and others. There are systems that do not have these features,
however, and Rust can work with those too! To do so, we tell Rust that we
don’t want to use the standard library via an attribute: `#![no_std]`.

> Note: This feature is technically stable, but there are some caveats. For
> one, you can build a `#![no_std]` _library_ on stable, but not a _binary_.
> For details on binaries without the standard library, see [the nightly
> chapter on `#![no_std]`](no-stdlib.html)

To use `#![no_std]`, add a it to your crate root:

```rust
#![no_std]

fn plus_one(x: i32) -> i32 {
    x + 1
}
```

Much of the functionality that’s exposed in the standard library is also
available via the [`core` crate](../core/). When we’re using the standard
library, Rust automatically brings `std` into scope, allowing you to use
its features without an explicit import. By the same token, when using
`!#[no_std]`, Rust will bring `core` into scope for you, as well as [its
prelude](../core/prelude/v1/). This means that a lot of code will Just Work:

```rust
#![no_std]

fn may_fail(failure: bool) -> Result<(), &'static str> {
    if failure {
        Err("this didn’t work!")
    } else {
        Ok(())
    }
}
```
