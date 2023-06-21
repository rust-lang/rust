// compile-flags: -Ztreat-err-as-bug
// dont-check-failure-status
// error-pattern: aborting due to `-Z treat-err-as-bug=1`
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
// rustc-env:RUST_BACKTRACE=0

fn main() {
    #[deny(while_true)]
    while true {}
}
