//@ unit-test: ElaborateDrops
//@ needs-unwind

#![feature(rustc_attrs, stmt_expr_attributes)]

// EMIT_MIR box_expr.main.ElaborateDrops.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK:   [[box:_.*]] = ShallowInitBox(
    // CHECK:   [[ptr:_.*]] = ((([[box]].0: std::ptr::Unique<S>).0: std::ptr::NonNull<S>).0: *const S);
    // CHECK:   (*[[ptr]]) = S::new() -> [return: [[ret:bb.*]], unwind: [[unwind:bb.*]]];
    // CHECK: [[ret]]: {
    // CHECK:   [[box2:_.*]] = move [[box]];
    // CHECK:   [[box3:_.*]] = move [[box2]];
    // CHECK:   std::mem::drop::<Box<S>>(move [[box3]])
    // CHECK: [[unwind]] (cleanup): {
    // CHECK:   [[boxref:_.*]] = &mut [[box]];
    // CHECK:   <Box<S> as Drop>::drop(move [[boxref]])

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
