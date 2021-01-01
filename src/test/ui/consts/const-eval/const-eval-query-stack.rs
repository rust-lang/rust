//~ERROR constructed but no error reported
// compile-flags: -Ztreat-err-as-bug=2
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
// normalize-stderr-test "\s\d{1,}: .*\n" -> ""
// normalize-stderr-test "\s at .*\n" -> ""
// normalize-stderr-test ".*note: Some details.*\n" -> ""

#![allow(unconditional_panic)]

#[warn(const_err)]
const X: i32 = 1 / 0; //~WARN any use of this value will cause an error

fn main() {
    let x: &'static i32 = &X;
    //~^ ERROR evaluation of constant expression failed
    println!("x={}", x);
}
