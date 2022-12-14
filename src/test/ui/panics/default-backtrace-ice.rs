// unset-rustc-env:RUST_BACKTRACE
// compile-flags:-Z treat-err-as-bug=1
// error-pattern:stack backtrace:
// failure-status:101
// normalize-stderr-test "note: .*" -> ""
// normalize-stderr-test "thread 'rustc' .*" -> ""
// normalize-stderr-test "  .*\n" -> ""

fn main() { missing_ident; }
