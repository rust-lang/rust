// compile-flags: -Zmir-opt-level=0


// EMIT_MIR core.ptr-drop_in_place.[String].AddMovesForPackedDrops.before.mir
fn main() {
    let _fn = std::ptr::drop_in_place::<[String]> as unsafe fn(_);
}
