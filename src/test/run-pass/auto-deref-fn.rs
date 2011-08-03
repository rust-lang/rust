fn add1(i: int) -> int { ret i + 1; }
fn main() {
    let f = @add1;
    let g = @f;
    let h = @@@add1;
    assert (f(5) == 6);
    assert (g(8) == 9);
    assert (h(0x1badd00d) == 0x1badd00e);
    assert ((@add1)(42) == 43);
}