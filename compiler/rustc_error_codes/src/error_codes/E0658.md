An unstable feature was used.

Erroneous code example:

```compile_fail,E0658
#[repr(u128)] // error: use of unstable library feature 'repr128'
enum Foo {
    Bar(u64),
}
```

If you're using a stable or a beta version of rustc, you won't be able to use
any unstable features. In order to do so, please switch to a nightly version of
rustc (by using [rustup]).

If you're using a nightly version of rustc, just add the corresponding feature
to be able to use it:

```
#![feature(repr128)]

#[repr(u128)] // ok!
enum Foo {
    Bar(u64),
}
```

[rustup]: https://rust-lang.github.io/rustup/concepts/channels.html
