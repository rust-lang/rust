//@ skip-filecheck
// Test that we don't ICE when trying to dump MIR for unusual item types and
// that we don't create filenames containing `<` and `>`
//@ compile-flags: -Zmir-opt-level=0

struct A;

// EMIT_MIR unusual_item_types.{impl#0}-ASSOCIATED_CONSTANT.built.after.mir
impl A {
    // The `|` makes it not a `trivial_const`, so it gets MIR built for it.
    const ASSOCIATED_CONSTANT: i32 = 2 | 0;
}

// See #59021
// EMIT_MIR unusual_item_types.Test-X-{constructor#0}.runtime-optimized.after.mir
enum Test {
    X(usize),
    Y { a: usize },
}

// EMIT_MIR unusual_item_types.E-V-{constant#0}.built.after.mir
enum E {
    // The `|` makes it not a `trivial_const`, so it gets MIR built for it.
    V = 5 | 0,
}

// EMIT_MIR core.ptr-drop_glue.Vec_i32_.AddMovesForPackedDrops.before.mir
pub fn main() {
    let f = Test::X as fn(usize) -> Test;
    let v = Vec::<i32>::new();
}
