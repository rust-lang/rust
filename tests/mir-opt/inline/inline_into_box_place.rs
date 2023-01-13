// ignore-endian-big
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -Z mir-opt-level=4

#![feature(box_syntax)]
// EMIT_MIR inline_into_box_place.main.Inline.diff
fn main() {
    let _x: Box<Vec<u32>> = box Vec::new();
}
