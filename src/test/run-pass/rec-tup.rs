// -*- rust -*-
type point = {x: int, y: int};

type rect = (point, point);

fn fst(r: &rect) -> point {
    let (fst, _) = r;
    ret fst;
}
fn snd(r: &rect) -> point {
    let (_, snd) = r;
    ret snd;
}

fn f(r: rect, x1: int, y1: int, x2: int, y2: int) {
    assert (fst(r).x == x1);
    assert (fst(r).y == y1);
    assert (snd(r).x == x2);
    assert (snd(r).y == y2);
}

fn main() {
    let r: rect = ({x: 10, y: 20}, {x: 11, y: 22});
    assert (fst(r).x == 10);
    assert (fst(r).y == 20);
    assert (snd(r).x == 11);
    assert (snd(r).y == 22);
    let r2 = r;
    let x: int = fst(r2).x;
    assert (x == 10);
    f(r, 10, 20, 11, 22);
    f(r2, 10, 20, 11, 22);
}
