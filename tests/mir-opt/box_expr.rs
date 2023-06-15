// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs, stmt_expr_attributes)]

// EMIT_MIR box_expr.main.ElaborateDrops.before.mir
fn main() {
    let x = #[rustc_box]
    Box::new(S::new());
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
