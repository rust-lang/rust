//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{LocalHandle, raw_handle};

struct Point {
    x: i32,
    y: i32,
}

macro_rules! place {
    ($p:expr) => {
        $p
    };
}

fn main() {
    let mut point = Point { x: 10, y: 20 };
    unsafe {
        let h: LocalHandle<i32> = raw_handle!(place!(point.y));
        *h.as_mut_ptr() = 999;
    }
    assert_eq!(point.x, 10);
    assert_eq!(point.y, 999);
}
