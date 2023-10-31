// ignore-endian-big
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// ignore-debug MIR alignment checks in std alter the diff, breaking the test
// compile-flags: -Zmir-opt-level=4 -Zinline-mir-hint-threshold=200

// EMIT_MIR inline_into_box_place.main.Inline.diff
fn main() {
    let _x: Box<Vec<u32>> = Box::new(Vec::new());
}
