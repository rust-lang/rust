//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;
use std::ops::place::{ProjectPlace, Subplace, raw_handle};

struct Point {
    x: i32,
    y: i32,
}

struct Line {
    start: Point,
    end: Point,
}

fn main() {
    // 1. Verify Subplace trait properties on field_of types
    let (offset, _) = Subplace::offset(<field_of!(Point, y)>::default(), ());
    assert_eq!(offset, std::mem::offset_of!(Point, y));

    // 2. Test place projection through LocalHandle
    let mut line = Line {
        start: Point { x: 10, y: 20 },
        end: Point { x: 30, y: 40 },
    };

    let line_handle = raw_handle!(line);

    // Project to `start`
    let start_subplace: field_of!(Line, start) = Default::default();
    let start_handle = unsafe { line_handle.project_place(start_subplace) };

    // Project to `start.y`
    let y_subplace: field_of!(Point, y) = Default::default();
    let y_handle = unsafe { start_handle.project_place(y_subplace) };

    unsafe {
        assert_eq!(*y_handle.as_ptr(), 20);
        *y_handle.as_mut_ptr() = 999;
    }

    assert_eq!(line.start.y, 999);

    // Project to `end.x`
    let end_subplace: field_of!(Line, end) = Default::default();
    let end_handle = unsafe { line_handle.project_place(end_subplace) };
    let x_subplace: field_of!(Point, x) = Default::default();
    let x_handle = unsafe { end_handle.project_place(x_subplace) };

    unsafe {
        assert_eq!(*x_handle.as_ptr(), 30);
        *x_handle.as_mut_ptr() = 777;
    }

    assert_eq!(line.end.x, 777);
}
