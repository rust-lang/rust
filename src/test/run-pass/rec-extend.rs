


// -*- rust -*-
type point = {x: int, y: int};

fn main() {
    let origin: point = {x: 0, y: 0};
    let right: point = {x: origin.x + 10,.. origin};
    let up: point = {y: origin.y + 10,.. origin};
    assert (origin.x == 0);
    assert (origin.y == 0);
    assert (right.x == 10);
    assert (right.y == 0);
    assert (up.x == 0);
    assert (up.y == 10);
}
