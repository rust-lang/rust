fn main() {
    fn f(i: uint) -> uint { i }
    let v = [-1f, 0f, 1f, 2f, 3f];
    let z = vec::foldl(f, v) { |x, _y| x } (22u);
    assert z == 22u;
}
