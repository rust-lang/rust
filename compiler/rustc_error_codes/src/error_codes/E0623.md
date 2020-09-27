A lifetime didn't match what was expected.

Erroneous code example:

```compile_fail,E0623
struct Foo<'a> {
    x: &'a isize,
}

fn bar<'short, 'long>(c: Foo<'short>, l: &'long isize) {
    let _: Foo<'long> = c; // error!
}
```

In this example, we tried to set a value with an incompatible lifetime to
another one (`'long` is unrelated to `'short`). We can solve this issue in
two different ways:

Either we make `'short` live at least as long as `'long`:

```
struct Foo<'a> {
    x: &'a isize,
}

// we set 'short to live at least as long as 'long
fn bar<'short: 'long, 'long>(c: Foo<'short>, l: &'long isize) {
    let _: Foo<'long> = c; // ok!
}
```

Or we use only one lifetime:

```
struct Foo<'a> {
    x: &'a isize,
}
fn bar<'short>(c: Foo<'short>, l: &'short isize) {
    let _: Foo<'short> = c; // ok!
}
```
