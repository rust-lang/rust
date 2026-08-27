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

struct PtrHolder<'a> {
    r: &'a mut i32,
    p: *mut i32,
}

fn main() {
    let mut line = Line { start: Point { x: 10, y: 20 }, end: Point { x: 30, y: 40 } };
    let mut val1 = 50;
    let mut val2 = 60;
    let holder = PtrHolder { r: &mut val1, p: &raw mut val2 };

    unsafe {
        let h: LocalHandle<i32> = raw_handle!(line.start.y);
        *h.as_mut_ptr() = 999;

        let line_ref = &mut line;
        let mut m: MutHandle<'_, i32> = raw_handle!(line_ref.start.x);
        *m.as_mut() = 111;

        let shared_ref = &line;
        let r: RefHandle<'_, i32> = raw_handle!(shared_ref.end.y);
        assert_eq!(*r.as_ref(), 40);

        let h_explicit: RefHandle<'_, i32> = raw_handle!((*shared_ref).end.y);
        assert_eq!(*h_explicit.as_ptr(), 40);

        let mut mr: MutHandle<'_, i32> = raw_handle!(*holder.r);
        *mr.as_mut() = 555;

        let mp: LocalHandle<i32> = raw_handle!(*holder.p);
        *mp.as_mut_ptr() = 666;
    }

    assert_eq!(line.start.x, 111);
    assert_eq!(line.start.y, 999);
    assert_eq!(val1, 555);
    assert_eq!(val2, 666);
}
