// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

#[derive(PartialOrd, PartialEq)]
pub struct MultiField(char, i16);

// Because this isn't derived by the impl, it's not on the `{impl#0}-partial_cmp`,
// and thus we need to call it to see what the inlined generic one produces.
pub fn demo_le(a: &MultiField, b: &MultiField) -> bool {
    *a <= *b
}

// EMIT_MIR derived_ord.{impl#0}-partial_cmp.PreCodegen.after.mir
// EMIT_MIR derived_ord.demo_le.PreCodegen.after.mir
