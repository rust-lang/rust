//@ test-mir-pass: ElaborateDrops
//@ needs-unwind
// skip-filecheck
#![feature(rustc_attrs, liballoc_internals)]

// EMIT_MIR box_expr.move_from_inner.ElaborateDrops.diff
fn move_from_inner() {
    let x = Box::new(S::new());
    drop(*x);
}

// EMIT_MIR box_expr.main.ElaborateDrops.diff
fn main() {
    let x = Box::new(S::new());
    drop(x);
}

struct S;

impl S {
    fn new() -> Self {
        S
    }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("splat!");
    }
}
