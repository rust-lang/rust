// compile-flags: --crate-type=lib -Copt-level=0 -Zmir-opt-level=1

// EMIT_MIR partial.{impl#1}-eq.LowerIntrinsics.diff
// EMIT_MIR partial.{impl#1}-eq.CopyPropagation.diff
// EMIT_MIR partial.{impl#1}-eq.PreCodegen.before.mir
// EMIT_MIR partial.{impl#2}-partial_cmp.LowerIntrinsics.diff
// EMIT_MIR partial.{impl#2}-partial_cmp.CopyPropagation.diff
// EMIT_MIR partial.{impl#2}-partial_cmp.PreCodegen.before.mir
#[derive(PartialEq, PartialOrd)]
pub enum E {
    A,
    B,
}
