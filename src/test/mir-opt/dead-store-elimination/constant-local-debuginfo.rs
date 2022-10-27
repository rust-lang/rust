// compile-flags: -Zmir-opt-level=2
// mir-opt-level is fixed at 2 because that's the max level that can be set on stable
// EMIT_MIR constant_local_debuginfo.main.DeadStoreElimination.diff
fn main() {
    let a = 1;
    let b = 4;

    foo(a + b);
}

#[inline(never)]
fn foo(x: i32) {
    std::process::exit(x);
}
