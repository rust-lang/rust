fn main() {
    fn f(i: fn() -> uint) -> uint { i() }
    let v = ~[-1f, 0f, 1f, 2f, 3f];
    let z = do do vec::foldl(f, v) { |x, _y| x } { || 22u };
    assert z == 22u;
}
