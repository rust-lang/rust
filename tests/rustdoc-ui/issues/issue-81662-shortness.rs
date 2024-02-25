//@ compile-flags:--test --error-format=short
//@ check-stdout
//@ error-pattern:cannot find function `foo` in this scope
//@ normalize-stdout-test: "tests/rustdoc-ui/issues" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

/// ```rust
/// foo();
/// ```
fn foo() {
    println!("Hello, world!");
}
