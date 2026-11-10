//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{LocalHandle, raw_handle};

struct Point {
    x: i32,
    y: i32,
}

static mut POINT: Point = Point { x: 10, y: 20 };

fn main() {
    let f = || unsafe {
        let h: LocalHandle<i32> = raw_handle!(POINT.y);
        *h.as_mut_ptr() = 999;
    };
    f();
    assert_eq!(unsafe { POINT.x }, 10);
    assert_eq!(unsafe { POINT.y }, 999);
}
