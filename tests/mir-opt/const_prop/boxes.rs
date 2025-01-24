//@ test-mir-pass: GVN
//@ compile-flags: -O
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs, liballoc_internals)]

// Note: this test verifies that we, in fact, do not const prop `#[rustc_box]`

// EMIT_MIR boxes.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: (*{{_.*}}) = const 42_i32;
    // CHECK: [[tmp:_.*]] = copy (*{{_.*}});
    // CHECK: [[x]] = copy [[tmp]];
    let x = *(std::boxed::box_new(42)) + 0;
}
