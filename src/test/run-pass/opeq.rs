


// -*- rust -*-
fn main() {
    let x: int = 1;
    x *= 2;
    log_full(core::debug, x);
    assert (x == 2);
    x += 3;
    log_full(core::debug, x);
    assert (x == 5);
    x *= x;
    log_full(core::debug, x);
    assert (x == 25);
    x /= 5;
    log_full(core::debug, x);
    assert (x == 5);
}
