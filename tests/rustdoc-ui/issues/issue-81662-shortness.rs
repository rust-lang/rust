// compile-flags:--test --error-format=short
// normalize-stdout-test: "tests/rustdoc-ui/issues" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// failure-status: 101

/// ```rust
/// foo();
/// ```
//~^^ ERROR cannot find function `foo` in this scope
fn foo() {
    println!("Hello, world!");
}
