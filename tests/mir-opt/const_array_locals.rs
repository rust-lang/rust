//@ test-mir-pass: GVN
#![feature(repr_simd)]

#[repr(simd)]
struct F32x8([f32; 8]);

// EMIT_MIR const_array_locals.main.GVN.diff
// CHECK-LABEL: fn main(
#[rustfmt::skip]
pub fn main() {
    let _arr = [255, 105, 15, 39, 62];
    // duplicate item
    let _barr = [255, 105, 15, 39, 62];
    let _foo = [
        [178, 9, 4, 56, 221],
        [193, 164, 194, 197, 6],
    ];
    let _darr = *&[255, 105, 15, 39, 62];

    consume([31, 96, 173, 50, 1]);

    let _f = F32x8([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 42.0]);

    // ice
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // 2D array
}

fn consume(_arr: [u32; 5]) {
    unimplemented!()
}
