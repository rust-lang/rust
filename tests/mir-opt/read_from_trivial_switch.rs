// Ensure that we don't optimize out `SwitchInt` reads even if that terminator
// branches to the same basic block on every target, since the operand may have
// side-effects that affect analysis of the MIR.
//
// See <https://github.com/rust-lang/miri/issues/4237>.

//@ test-mir-pass: SimplifyCfg-initial
//@ compile-flags: -Zmir-preserve-ub

// EMIT_MIR read_from_trivial_switch.main.SimplifyCfg-initial.diff
fn main() {
    let ref_ = &1i32;
    // CHECK: switchInt
    let &(0 | _) = ref_;
}
