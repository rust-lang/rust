#### Note: this error code is no longer emitted by the compiler.

You tried to supply a type which doesn't implement some trait in a location
which expected that trait. This error typically occurs when working with
`Fn`-based types. Erroneous code example:

```compile_fail
fn foo<F: Fn(usize)>(x: F) { }

fn main() {
    // type mismatch: ... implements the trait `core::ops::Fn<(String,)>`,
    // but the trait `core::ops::Fn<(usize,)>` is required
    // [E0281]
    foo(|y: String| { });
}
```

The issue in this case is that `foo` is defined as accepting a `Fn` with one
argument of type `String`, but the closure we attempted to pass to it requires
one arguments of type `usize`.
