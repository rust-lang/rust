# `coverage_attribute`

The tracking issue for this feature is: [#84605]

[#84605]: https://github.com/rust-lang/rust/issues/84605

---

The `coverage` attribute can be used to selectively disable coverage
instrumentation in an annotated function. This might be useful to:

-   Avoid instrumentation overhead in a performance critical function
-   Avoid generating coverage for a function that is not meant to be executed,
    but still target 100% coverage for the rest of the program.

## Example

```rust
#![feature(coverage_attribute)]

// `foo()` will get coverage instrumentation (by default)
fn foo() {
  // ...
}

#[coverage(off)]
fn bar() {
  // ...
}
```
