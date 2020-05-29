// compile-flags: -Zmir-opt-level=0

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.ptr-drop_in_place.[std__string__String].AddMovesForPackedDrops.before.mir
fn main() {
    let _fn = std::ptr::drop_in_place::<[String]> as unsafe fn(_);
}
