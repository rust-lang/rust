


// -*- rust -*-
type point = rec(int x, int y);

fn main() {
    let point origin = rec(x=0, y=0);
    let point right = rec(x=origin.x + 10 with origin);
    let point up = rec(y=origin.y + 10 with origin);
    assert (origin.x == 0);
    assert (origin.y == 0);
    assert (right.x == 10);
    assert (right.y == 0);
    assert (up.x == 0);
    assert (up.y == 10);
}