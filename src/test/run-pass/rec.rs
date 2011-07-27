


// -*- rust -*-
type rect = {x: int, y: int, w: int, h: int};

fn f(r: rect, x: int, y: int, w: int, h: int) {
    assert (r.x == x);
    assert (r.y == y);
    assert (r.w == w);
    assert (r.h == h);
}

fn main() {
    let r: rect = {x: 10, y: 20, w: 100, h: 200};
    assert (r.x == 10);
    assert (r.y == 20);
    assert (r.w == 100);
    assert (r.h == 200);
    let r2: rect = r;
    let x: int = r2.x;
    assert (x == 10);
    f(r, 10, 20, 100, 200);
    f(r2, 10, 20, 100, 200);
}