// Issue #51162: A failed doctest was not printing its stdout/stderr
// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// failure-status: 101

// doctest fails at runtime
/// ```
/// println!("stdout 1");
/// eprintln!("stderr 1");
/// println!("stdout 2");
/// eprintln!("stderr 2");
/// panic!("oh no");
/// ```
pub struct SomeStruct;

// doctest fails at compile time
/// ```
/// no
/// ```
pub struct OtherStruct;
