//@ compile-flags: -Ztreat-err-as-bug
//@ failure-status: 101
//@ error-pattern: aborting due to `-Z treat-err-as-bug=1`
//@ error-pattern: [eval_static_initializer] evaluating initializer of static `C`
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc' panicked.*:\n.*\n" -> ""
//@ rustc-env:RUST_BACKTRACE=0

#![crate_type = "rlib"]

pub static C: u32 = 0 - 1;
//~^ ERROR could not evaluate static initializer
