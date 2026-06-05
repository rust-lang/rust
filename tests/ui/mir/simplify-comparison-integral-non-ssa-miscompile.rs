// Regression test for a miscompile in SimplifyComparisonIntegral: it rewrote a
// switchInt on a non-SSA comparison local, dropping a later reassignment, after
// DestinationPropagation was enabled by default. See rust-lang/rust#150904.

//@ run-pass
//@ compile-flags: -O

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

fn main() {
    assert_eq!(run_my_check(42, 0), 1);
    assert_eq!(run_my_check(42, 1), 0);
    assert_eq!(run_my_check(0, 0), 0);
}
