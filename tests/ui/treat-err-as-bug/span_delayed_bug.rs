// compile-flags: -Ztreat-err-as-bug
// failure-status: 101
// error-pattern: aborting due to `-Z treat-err-as-bug=1`
// error-pattern: [trigger_span_delayed_bug] triggering a span delayed bug for testing incremental
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*:\n.*\n" -> ""
// rustc-env:RUST_BACKTRACE=0

#![feature(rustc_attrs)]

#[rustc_error(span_delayed_bug_from_inside_query)]
fn main() {}
