`impl Trait` is not allowed in path parameters.

Erroneous code example:

```compile_fail,E0667
fn some_fn(mut x: impl Iterator) -> <impl Iterator>::Item { // error!
    x.next().unwrap()
}
```

You cannot use `impl Trait` in path parameters. If you want something
equivalent, you can do this instead:

```
fn some_fn<T: Iterator>(mut x: T) -> T::Item { // ok!
    x.next().unwrap()
}
```
