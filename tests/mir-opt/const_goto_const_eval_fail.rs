// skip-filecheck
#![feature(min_const_generics)]
#![crate_type = "lib"]

//@ compile-flags: -Zunsound-mir-opts

// If const eval fails, then don't crash
// EMIT_MIR const_goto_const_eval_fail.f.JumpThreading.diff
pub fn f<const A: i32, const B: bool>() -> u64 {
    match {
        match A {
            1 | 2 | 3 => B,
            _ => true,
        }
    } {
        false => 1,
        true => 2,
    }
}
