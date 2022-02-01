// This test that no matter the optimization level or the target feature enable, the non
// aggregation of homogenous floats in the abi is sound and still produce the right answer.

// revisions: opt-0 opt-0-native opt-1 opt-1-native opt-2 opt-2-native opt-3 opt-3-native
// [opt-0]: compile-flags: -C opt-level=0
// [opt-1]: compile-flags: -C opt-level=1
// [opt-2]: compile-flags: -C opt-level=2
// [opt-3]: compile-flags: -C opt-level=3
// [opt-0-native]: compile-flags: -C target-cpu=native
// [opt-1-native]: compile-flags: -C target-cpu=native
// [opt-2-native]: compile-flags: -C target-cpu=native
// [opt-3-native]: compile-flags: -C target-cpu=native
// run-pass

#![feature(core_intrinsics)]

use std::intrinsics::black_box;

pub fn sum_f32(a: f32, b: f32) -> f32 {
    a + b
}

pub fn sum_f32x2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

pub fn sum_f32x3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sum_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn main() {
    assert_eq!(1., black_box(sum_f32(black_box(0.), black_box(1.))));
    assert_eq!([2., 2.], black_box(sum_f32x2(black_box([2., 0.]), black_box([0., 2.]))));
    assert_eq!(
        [3., 3., 3.],
        black_box(sum_f32x3(black_box([1., 2., 3.]), black_box([2., 1., 0.])))
    );
    assert_eq!(
        [4., 4., 4., 4.],
        black_box(sum_f32x4(black_box([1., 2., 3., 4.]), black_box([3., 2., 1., 0.])))
    );
}
