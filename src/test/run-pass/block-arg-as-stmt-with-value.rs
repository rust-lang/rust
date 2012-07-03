
fn compute1() -> float {
    let v = ~[0f, 1f, 2f, 3f];

    // This is actually a (block-style) statement followed by
    // a unary tail expression
    do vec::foldl(0f, v) |x, y| { x + y } - 10f
}

fn main() {
    let x = compute1();
    log(debug, x);
    assert(x == -10f);
}
