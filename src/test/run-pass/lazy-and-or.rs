

fn incr(&mutable int x) -> bool { x += 1; assert (false); ret false; }

fn main() {
    auto x = 1 == 2 || 3 == 3;
    assert (x);
    let int y = 10;
    log x || incr(y);
    assert (y == 10);
    if (true && x) { assert (true); } else { assert (false); }
}