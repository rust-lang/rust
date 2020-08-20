// compile-flags: -Ztreat-err-as-bug
// build-fail
// failure-status: 101
// rustc-env:RUST_BACKTRACE=1
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "thread.*panicked.*\n" -> ""
// normalize-stderr-test "stack backtrace:\n" -> ""
// normalize-stderr-test "  \d{1,}: .*\n" -> ""
// normalize-stderr-test ".*note: Some details.*\n" -> ""

#![allow(unconditional_panic)]

fn main() {
    let x: &'static i32 = &(1 / 0);
    //~^ ERROR reaching this expression at runtime will panic or abort [const_err]
    println!("x={}", x);
}
