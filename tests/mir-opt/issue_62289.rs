// check that we don't forget to drop the Box if we early return before
// initializing it
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(box_syntax)]

// EMIT_MIR issue_62289.test.ElaborateDrops.before.mir
fn test() -> Option<Box<u32>> {
    Some(box (None?))
}

fn main() {
    test();
}
