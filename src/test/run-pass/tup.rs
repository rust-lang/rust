


// -*- rust -*-
type point = tup(int, int);

fn f(point p, int x, int y) { assert (p._0 == x); assert (p._1 == y); }

fn main() {
    let point p = tup(10, 20);
    assert (p._0 == 10);
    assert (p._1 == 20);
    let point p2 = p;
    let int x = p2._0;
    assert (x == 10);
    f(p, 10, 20);
    f(p2, 10, 20);
}