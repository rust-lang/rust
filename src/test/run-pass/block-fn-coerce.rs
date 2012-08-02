fn force(f: fn() -> int) -> int { return f(); }
fn main() {
    fn f() -> int { return 7; }
    assert (force(f) == 7);
    let g = {||force(f)};
    assert (g() == 7);
}
