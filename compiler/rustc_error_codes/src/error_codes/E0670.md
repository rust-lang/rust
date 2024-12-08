Rust 2015 does not permit the use of `async fn`.

Erroneous code example:

```compile_fail,E0670
async fn foo() {}
```

Switch to the Rust 2018 edition to use `async fn`.
