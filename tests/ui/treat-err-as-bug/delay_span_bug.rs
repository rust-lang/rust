// compile-flags: -Ztreat-err-as-bug
// rustc-env:RUSTC_ICE=0
// failure-status: 101
// error-pattern: aborting due to `-Z treat-err-as-bug=1`
// error-pattern: [trigger_delay_span_bug] triggering a delay span bug
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
// rustc-env:RUST_BACKTRACE=0

#![feature(rustc_attrs)]

#[rustc_error(delay_span_bug_from_inside_query)]
fn main() {}
