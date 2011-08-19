


// -*- rust -*-
type point = {x: int, y: int, mutable z: int};

fn f(p: @point) { assert (p.z == 12); p.z = 13; assert (p.z == 13); }

fn main() {
    let a: point = {x: 10, y: 11, mutable z: 12};
    let b: @point = @a;
    assert (b.z == 12);
    f(b);
    assert (a.z == 12);
    assert (b.z == 13);
}
