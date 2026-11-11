#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[repr(C)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_struct(point: &Point) -> f64 {
    point.x + point.y * 2.0 + point.z * 3.0
}

fn main() {
    let point = Point { x: 1.0, y: 2.0, z: 3.0 };
    let mut d_point = Point { x: 0.0, y: 0.0, z: 0.0 };
    let _result = d_test(&point, &mut d_point, 1.0);
}
