


// -*- rust -*-
type point = rec(int x, int y, mutable int z);

fn f(@point p) { assert (p.z == 12); p.z = 13; assert (p.z == 13); }

fn main() {
    let point a = rec(x=10, y=11, mutable z=12);
    let @point b = @a;
    assert (b.z == 12);
    f(b);
    assert (a.z == 12);
    assert (b.z == 13);
}