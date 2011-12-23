

fn incr(&x: int) -> bool { x += 1; assert (false); ret false; }

fn main() {
    let x = 1 == 2 || 3 == 3;
    assert (x);
    let y: int = 10;
    log(debug, x || incr(y));
    assert (y == 10);
    if true && x { assert (true); } else { assert (false); }
}
