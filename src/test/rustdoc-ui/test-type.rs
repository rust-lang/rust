// compile-flags: --test --test-args=--test-threads=1
// check-pass
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// let a = true;
/// ```
/// ```should_panic
/// panic!()
/// ```
/// ```ignore (incomplete-code)
/// fn foo() {
/// ```
/// ```no_run
/// loop {
///     println!("Hello, world");
/// }
/// ```
/// fails to compile
/// ```compile_fail
/// let x = 5;
/// x += 2; // shouldn't compile!
/// ```

pub fn f() {}
