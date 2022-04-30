// check-pass

// compile-flags:--test --test-args=--include-ignored --test-args=--test-threads=1
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// These code blocks should be treated as rust code in documentation, but never
/// treated as a test and compiled, even when testing ignored tests.
///
/// ```rust,no_compile
/// // haha rustc
/// i"have" 0utsmarted k#thee
/// ```
///
/// ```no_compile
/// // haha rustc
/// i"have" 0utsmarted k#thee
/// ```
///
/// For comparison, ignore does get run:
///
/// ```ignore (but-actually-we-include-ignored)
/// println!("Hello, world");
/// ```
struct S;
