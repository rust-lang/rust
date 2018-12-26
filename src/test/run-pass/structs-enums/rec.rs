// run-pass

#[derive(Copy, Clone)]
struct Rect {x: isize, y: isize, w: isize, h: isize}

fn f(r: Rect, x: isize, y: isize, w: isize, h: isize) {
    assert_eq!(r.x, x);
    assert_eq!(r.y, y);
    assert_eq!(r.w, w);
    assert_eq!(r.h, h);
}

pub fn main() {
    let r: Rect = Rect {x: 10, y: 20, w: 100, h: 200};
    assert_eq!(r.x, 10);
    assert_eq!(r.y, 20);
    assert_eq!(r.w, 100);
    assert_eq!(r.h, 200);
    let r2: Rect = r;
    let x: isize = r2.x;
    assert_eq!(x, 10);
    f(r, 10, 20, 100, 200);
    f(r2, 10, 20, 100, 200);
}
