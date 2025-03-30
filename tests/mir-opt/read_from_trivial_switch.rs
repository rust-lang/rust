//@ test-mir-pass: SimplifyCfg-initial
//@ compile-flags: -Zmir-preserve-ub

// EMIT_MIR read_from_trivial_switch.main.SimplifyCfg-initial.diff
fn main() {
    let ref_ = &1i32;
    // CHECK: switchInt
    let &(0 | _) = ref_;
}
