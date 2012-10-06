

fn incr(x: &mut int) -> bool { *x += 1; assert (false); return false; }

fn main() {
    let x = 1 == 2 || 3 == 3;
    assert (x);
    let mut y: int = 10;
    log(debug, x || incr(&mut y));
    assert (y == 10);
    if true && x { assert (true); } else { assert (false); }
}
