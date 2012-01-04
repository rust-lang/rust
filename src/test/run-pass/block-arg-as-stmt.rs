
fn compute1() -> float {
    let v = [0f, 1f, 2f, 3f];

    // Here the "-10f" parses as a second
    // statement in tail position:
    vec::foldl(0f, v) { |x, y| x + y } - 10f
}

fn compute2() -> float {
    let v = [0f, 1f, 2f, 3f];

    // Here the ret makes this explicit:
    ret vec::foldl(0f, v) { |x, y| x + y } - 10f;
}

fn main() {
    let x = compute1();
    log(debug, x);
    assert(x == -10f);

    let y = compute2();
    log(debug, y);
    assert(y == -4f);
}
