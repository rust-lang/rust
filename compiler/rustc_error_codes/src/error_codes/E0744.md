An unsupported expression was used inside a const context.

Erroneous code example:

```compile_fail,edition2018,E0744
const _: i32 = {
    async { 0 }.await
};
```

At the moment, `.await` is forbidden inside a `const`, `static`, or `const fn`.

This may be allowed at some point in the future, but the implementation is not
yet complete. See the tracking issue for [`async`] in `const fn`.

[`async`]: https://github.com/rust-lang/rust/issues/69431
