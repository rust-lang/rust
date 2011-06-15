


// -*- rust -*-
type point = rec(int x, int y);

type rect = tup(point, point);

fn f(rect r, int x1, int y1, int x2, int y2) {
    assert (r._0.x == x1);
    assert (r._0.y == y1);
    assert (r._1.x == x2);
    assert (r._1.y == y2);
}

fn main() {
    let rect r = tup(rec(x=10, y=20), rec(x=11, y=22));
    assert (r._0.x == 10);
    assert (r._0.y == 20);
    assert (r._1.x == 11);
    assert (r._1.y == 22);
    let rect r2 = r;
    let int x = r2._0.x;
    assert (x == 10);
    f(r, 10, 20, 11, 22);
    f(r2, 10, 20, 11, 22);
}