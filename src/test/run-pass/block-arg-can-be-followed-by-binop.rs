fn main() {
    let v = [-1f, 0f, 1f, 2f, 3f];

    // Trailing expressions require parentheses:
    let y = vec::foldl(0f, v) { |x, y| x + y } + 10f;

    assert y == 15f;
}
