//@ compile-flags: -Ztreat-err-as-bug=1
//@ failure-status: 101
//@ rustc-env:RUST_BACKTRACE=1
//@ normalize-stderr: "\nerror: .*unexpectedly panicked.*\n\n" -> ""
//@ normalize-stderr: "note: we would appreciate a bug report.*\n\n" -> ""
//@ normalize-stderr: "note: compiler flags.*\n\n" -> ""
//@ normalize-stderr: "note: rustc.*running on.*\n\n" -> ""
//@ normalize-stderr: "thread.*panicked.*:\n.*\n" -> ""
//@ normalize-stderr: "stack backtrace:\n" -> ""
//@ normalize-stderr: "\s\d{1,}: .*\n" -> ""
//@ normalize-stderr: "\s at .*\n" -> ""
//@ normalize-stderr: ".*note: Some details.*\n" -> ""
//@ normalize-stderr: ".*omitted \d{1,} frame.*\n" -> ""
#![allow(unconditional_panic)]

const X: i32 = 1 / 0; //~ERROR attempt to divide `1_i32` by zero

fn main() {
    let x: &'static i32 = &X;
    println!("x={}", x);
}
