// Tests for the new |args| expr lambda syntax

fn f(i: int, f: fn(int) -> int) -> int { f(i) }

fn g(g: fn()) { }

fn ff() -> fn@(int) -> int {
    return |x| x + 1;
}

fn main() {
    assert f(10, |a| a) == 10;
    g(||());
    assert do f(10) |a| { a } == 10;
    do g() { }
    let _x: fn@() -> int = || 10;
    let _y: fn@(int) -> int = |a| a;
    assert ff()(10) == 11;
}
