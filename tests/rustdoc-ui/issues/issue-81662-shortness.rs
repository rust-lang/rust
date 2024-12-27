//@ compile-flags:--test --error-format=short
//@ check-stdout
//@ error-pattern:cannot find function `foo`
//@ normalize-stdout: "tests/rustdoc-ui/issues" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

/// ```rust
/// foo();
/// ```
fn foo() {
    println!("Hello, world!");
}
