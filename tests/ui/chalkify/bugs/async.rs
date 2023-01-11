// edition:2021
// known-bug
// unset-rustc-env:RUST_BACKTRACE
// compile-flags:-Z trait-solver=chalk
// error-pattern:stack backtrace:
// failure-status:101
// normalize-stderr-test "note: .*" -> ""
// normalize-stderr-test "thread 'rustc' .*" -> ""
// normalize-stderr-test " +[0-9]+:.*\n" -> ""
// normalize-stderr-test " +at .*\n" -> ""

fn main() -> () {}

async fn foo(x: u32) -> u32 {
    x
}
