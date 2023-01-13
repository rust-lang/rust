// check-fail
// known-bug
// unset-rustc-env:RUST_BACKTRACE
// compile-flags:-Z trait-solver=chalk --edition=2021
// error-pattern:stack backtrace:
// failure-status:101
// normalize-stderr-test "note: .*" -> ""
// normalize-stderr-test "thread 'rustc' .*" -> ""
// normalize-stderr-test "  .*\n" -> ""
// normalize-stderr-test "DefId([^)]*)" -> "..."

fn main() -> () {}

async fn foo(x: u32) -> u32 {
    x
}
