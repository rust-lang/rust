// Regression test for <https://github.com/rust-lang/rust/issues/134221>.
// It checks that even if there are comments in the attributes, the attributes
// will still be generated correctly (and therefore fail in this test).

//@ compile-flags:--test --test-args --test-threads=1
//@ failure-status: 101
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

/*!
```rust
#![feature(
  foo, //
)]
```

```rust
#![feature(
  foo,
)]
```

```rust
#![
```
*/
