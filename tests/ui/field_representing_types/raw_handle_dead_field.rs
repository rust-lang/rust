//@ check-pass
#![expect(incomplete_features)]
#![feature(field_projections)]
#![deny(dead_code)]

use std::ops::place::{MutHandle, raw_handle};

pub struct Point {
    pub x: i32,
    y: i32,
}

pub fn read_y(point: &mut Point) -> i32 {
    unsafe {
        let h: MutHandle<'_, i32> = raw_handle!(point.y);
        *h.as_ref()
    }
}

pub fn new() -> Point {
    Point { x: 10, y: 20 }
}

fn main() {
    let mut point = new();
    let _ = point.x + read_y(&mut point);
}
