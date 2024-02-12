// compile-flags: -Ztreat-err-as-bug -Zeagerly-emit-delayed-bugs
// failure-status: 101
// error-pattern: aborting due to `-Z treat-err-as-bug=1`
// error-pattern: [trigger_delayed_bug] triggering a delayed bug for testing incremental
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*:\n.*\n" -> ""
// rustc-env:RUST_BACKTRACE=0

#![feature(rustc_attrs)]

#[rustc_error(delayed_bug_from_inside_query)]
fn main() {}
