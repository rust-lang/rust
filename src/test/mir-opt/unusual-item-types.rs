// Test that we don't ICE when trying to dump MIR for unusual item types and
// that we don't create filenames containing `<` and `>`
// compile-flags: -Zmir-opt-level=0


struct A;

// EMIT_MIR unusual_item_types.{impl#0}-ASSOCIATED_CONSTANT.mir_map.0.mir
impl A {
    const ASSOCIATED_CONSTANT: i32 = 2;
}

// See #59021
// EMIT_MIR unusual_item_types.Test-X-{constructor#0}.mir_map.0.mir
enum Test {
    X(usize),
    Y { a: usize },
}

// EMIT_MIR unusual_item_types.E-V-{constant#0}.mir_map.0.mir
enum E {
    V = 5,
}

fn main() {
    let f = Test::X as fn(usize) -> Test;
// EMIT_MIR core.ptr-drop_in_place.Vec_i32_.AddMovesForPackedDrops.before.mir
    let v = Vec::<i32>::new();
}
