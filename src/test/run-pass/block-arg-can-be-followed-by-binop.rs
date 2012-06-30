fn main() {
    let v = ~[-1f, 0f, 1f, 2f, 3f];

    // Trailing expressions don't require parentheses:
    let y = do vec::foldl(0f, v) |x, y| { x + y } + 10f;

    assert y == 15f;
}
