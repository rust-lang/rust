// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -Z mir-opt-level=3
// only-64bit FIXME: the mir representation of RawVec depends on ptr size
#![feature(box_syntax)]

// EMIT_MIR rustc.main.Inline.diff
fn main() {
    let _x: Box<Vec<u32>> = box Vec::new();
}
