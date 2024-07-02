//@ test-mir-pass: GVN
#![feature(repr_simd)]

#[repr(simd)]
struct F32x8([f32; 8]);

// EMIT_MIR const_array_locals.main.GVN.diff
// CHECK-LABEL: fn main(
// CHECK: debug _arr => [[_arr:_[0-9]+]];
// CHECK: debug _barr => [[_barr:_[0-9]+]];
// CHECK: debug _foo => [[_foo:_[0-9]+]];
// CHECK: debug _darr => [[_darr:_[0-9]+]];
// CHECK: debug _f => [[_f:_[0-9]+]];
pub fn main() {
    // CHECK: [[_arr]] = const [255_i32, 105_i32, 15_i32, 39_i32, 62_i32];
    let _arr = [255, 105, 15, 39, 62];
    // duplicate item
    // CHECK: [[_barr]] = [[_arr]];
    let _barr = [255, 105, 15, 39, 62];
    // CHECK: [[subarray1:_[0-9]+]] = const [178_i32, 9_i32, 4_i32, 56_i32, 221_i32];
    // CHECK: [[subarray2:_[0-9]+]] = const [193_i32, 164_i32, 194_i32, 197_i32, 6_i32];
    // CHECK: [[_foo]] = [move [[subarray1]], move [[subarray2]]];
    let _foo = [[178, 9, 4, 56, 221], [193, 164, 194, 197, 6]];
    // CHECK: [[PROMOTED:_[0-9]+]] = const main::promoted[0];
    // CHECK: [[_darr]] = (*[[PROMOTED]]);
    let _darr = *&[255, 105, 15, 39, 62];

    // CHECK: [[ARG:_[0-9]+]] = const [31_u32, 96_u32, 173_u32, 50_u32, 1_u32];
    // CHECK: consume(move [[ARG]])
    consume([31, 96, 173, 50, 1]);

    // CHECK: [[OP:_[0-9]+]] = const [1f32, 2f32, 3f32, 1f32, 1f32, 1f32, 1f32, 42f32];
    // CHECK: [[_f]] = F32x8(move [[OP]]);
    let _f = F32x8([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 42.0]);

    // ice with small arrays
    // CHECK: [[A:_[0-9]+]] = [const 1_i32, const 0_i32, const 0_i32];
    // CHECK: [[B:_[0-9]+]] = [const 0_i32, const 1_i32, const 0_i32];
    // CHECK: [[C:_[0-9]+]] = [const 0_i32, const 0_i32, const 1_i32];
    // CHECK: {{_[0-9]+}} = [move [[A]], move [[B]], move [[C]]];
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // 2D array
}

fn consume(_arr: [u32; 5]) {
    unimplemented!()
}
