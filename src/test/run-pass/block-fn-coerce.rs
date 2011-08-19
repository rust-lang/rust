fn force(f: &block() -> int) -> int { ret f(); }
fn main() {
    let f = fn () -> int { ret 7 };
    assert (force(f) == 7);
    let g = bind force(f);
    assert (g() == 7);
}
