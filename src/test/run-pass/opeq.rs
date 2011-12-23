


// -*- rust -*-
fn main() {
    let x: int = 1;
    x *= 2;
    log(debug, x);
    assert (x == 2);
    x += 3;
    log(debug, x);
    assert (x == 5);
    x *= x;
    log(debug, x);
    assert (x == 25);
    x /= 5;
    log(debug, x);
    assert (x == 5);
}
