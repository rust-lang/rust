This error indicates a type mismatch in closure arguments.

Erroneous code example:

```compile_fail,E0631
fn foo<F: Fn(i32)>(f: F) {
}

fn main() {
    foo(|x: &str| {});
}
```

The error occurs because `foo` accepts a closure that takes an `i32` argument,
but in `main`, it is passed a closure with a `&str` argument.

This can be resolved by changing the type annotation or removing it entirely
if it can be inferred.

```
fn foo<F: Fn(i32)>(f: F) {
}

fn main() {
    foo(|x: i32| {});
}
```
