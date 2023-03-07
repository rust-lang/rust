// ignore-wasm32-bare compiled with panic=abort by default

#![feature(box_syntax)]

// EMIT_MIR box_expr.main.ElaborateDrops.before.mir
fn main() {
    let x = box S::new();
    drop(x);
}

struct S;

impl S {
    fn new() -> Self { S }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("splat!");
    }
}
