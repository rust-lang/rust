// skip-filecheck
//@ compile-flags: -Zmir-opt-level=0 -Clink-dead-code
// mir-opt tests are always built as rlibs so that they seamlessly cross-compile,
// so this test only produces MIR for the drop_in_place we're looking for
// if we use -Clink-dead-code.

// EMIT_MIR core.ptr-drop_in_place.[String].AddMovesForPackedDrops.before.mir
// EMIT_MIR core.ptr-drop_in_place.[String;42].AddMovesForPackedDrops.before.mir
fn main() {
    let _fn = std::ptr::drop_in_place::<[String]> as unsafe fn(_);
    let _fn = std::ptr::drop_in_place::<[String; 42]> as unsafe fn(_);
}
