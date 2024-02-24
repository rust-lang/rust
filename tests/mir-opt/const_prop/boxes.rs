//@ unit-test: GVN
//@ compile-flags: -O
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs, stmt_expr_attributes)]

// Note: this test verifies that we, in fact, do not const prop `#[rustc_box]`

// EMIT_MIR boxes.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: (*{{_.*}}) = const 42_i32;
    // CHECK: [[tmp:_.*]] = (*{{_.*}});
    // CHECK: [[x]] = [[tmp]];
    let x = *(#[rustc_box]
    Box::new(42))
        + 0;
}
