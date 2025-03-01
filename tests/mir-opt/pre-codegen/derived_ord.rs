// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

#[derive(PartialOrd, PartialEq, Ord, Eq)]
pub struct MultiField(char, i16);

// EMIT_MIR derived_ord.{impl#0}-partial_cmp.PreCodegen.after.mir
// EMIT_MIR derived_ord.{impl#3}-cmp.PreCodegen.after.mir
