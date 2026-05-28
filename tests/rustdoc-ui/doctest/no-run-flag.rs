// test the behavior of the --no-run flag

//@ check-pass
//@ compile-flags:-Z unstable-options --test --no-run --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

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
/// Ok the test does not run
/// ```
/// panic!()
/// ```
/// Ok the test does not run
/// ```should_panic
/// loop {
///     println!("Hello, world");
/// panic!()
/// }
/// ```
pub fn f() {}
