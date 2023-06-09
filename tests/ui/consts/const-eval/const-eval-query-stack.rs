// compile-flags: -Ztreat-err-as-bug=1
// failure-status: 101
// rustc-env:RUST_BACKTRACE=1
// normalize-stderr-test "\nerror: .*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "thread.*panicked.*\n" -> ""
// normalize-stderr-test "stack backtrace:\n" -> ""
// normalize-stderr-test "\s\d{1,}: .*\n" -> ""
// normalize-stderr-test "\s at .*\n" -> ""
// normalize-stderr-test ".*note: Some details.*\n" -> ""

#![allow(unconditional_panic)]

const X: i32 = 1 / 0; //~ERROR constant

fn main() {
    let x: &'static i32 = &X;
    println!("x={}", x);
}
