// skip-filecheck
#![crate_type = "lib"]

// Regression test for <https://github.com/rust-lang/rust/issues/150904>.

// EMIT_MIR cmp_switch.run_my_check.PreCodegen.after.mir
#[inline(never)]
fn run_my_check(v0: u64, v1: u64) -> i32 {
    if do_check(v0, v1) { 1 } else { 0 }
}

#[inline(always)]
fn do_check(v0: u64, v1: u64) -> bool {
    let mut ok = v0 == 42;
    ok &= v1 == 0;
    ok
}
