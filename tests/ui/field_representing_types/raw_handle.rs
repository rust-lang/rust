//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{LocalHandle, MutHandle, RefHandle, raw_handle};

struct Point {
    x: i32,
    y: i32,
}

struct Line {
    start: Point,
    end: Point,
}

fn main() {
    let mut line = Line { start: Point { x: 10, y: 20 }, end: Point { x: 30, y: 40 } };

    unsafe {
        let h: LocalHandle<i32> = raw_handle!(line.start.y);
        *h.as_mut_ptr() = 999;

        let line_ref = &mut line;
        let mut m: MutHandle<'_, i32> = raw_handle!(line_ref.start.x);
        *m.as_mut() = 111;

        let shared_ref = &line;
        let r: RefHandle<'_, i32> = raw_handle!((*shared_ref).end.y);
        assert_eq!(*r.as_ref(), 40);
    }

    assert_eq!(line.start.x, 111);
    assert_eq!(line.start.y, 999);
}
