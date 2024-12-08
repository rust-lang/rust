// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

#[derive(PartialOrd, PartialEq)]
pub struct MultiField(char, i16);

// EMIT_MIR derived_ord.{impl#0}-partial_cmp.PreCodegen.after.mir
