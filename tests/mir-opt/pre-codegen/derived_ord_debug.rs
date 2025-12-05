//@ compile-flags: -Copt-level=0 -Zmir-opt-level=1 -Cdebuginfo=limited
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

#[derive(PartialOrd, Ord, PartialEq, Eq)]
pub struct MultiField(char, i16);

// EMIT_MIR derived_ord_debug.{impl#0}-partial_cmp.PreCodegen.after.mir
// EMIT_MIR derived_ord_debug.{impl#1}-cmp.PreCodegen.after.mir

// CHECK-LABEL: partial_cmp(_1: &MultiField, _2: &MultiField) -> Option<std::cmp::Ordering>
// CHECK: = <char as PartialOrd>::partial_cmp(
// CHECK: = <i16 as PartialOrd>::partial_cmp(

// CHECK-LABEL: cmp(_1: &MultiField, _2: &MultiField) -> std::cmp::Ordering
// CHECK: = <char as Ord>::cmp(
// CHECK: = <i16 as Ord>::cmp(
