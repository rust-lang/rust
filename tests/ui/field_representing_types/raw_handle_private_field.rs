#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{LocalHandle, raw_handle};

mod m {
    pub struct Point {
        pub x: i32,
        y: i32,
    }

    impl Point {
        pub fn new() -> Point {
            Point { x: 10, y: 20 }
        }
    }
}

fn main() {
    let mut point = m::Point::new();
    unsafe {
        let h: LocalHandle<i32> = raw_handle!(point.y);
        //~^ ERROR: field `y` of struct `Point` is private [E0616]
        *h.as_mut_ptr() = 999;
    }
    assert_eq!(point.x, 10);
}
