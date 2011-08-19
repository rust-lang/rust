


// -*- rust -*-
fn main() {
    let x: int = 15;
    let y: int = 5;
    assert (x / 5 == 3);
    assert (x / 4 == 3);
    assert (x / 3 == 5);
    assert (x / y == 3);
    assert (15 / y == 3);
    assert (x % 5 == 0);
    assert (x % 4 == 3);
    assert (x % 3 == 0);
    assert (x % y == 0);
    assert (15 % y == 0);
}
