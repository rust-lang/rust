fn main() {
    #macro([#apply(f, [x, ...]), f(x, ...)]);

    fn add(a: int, b: int) -> int { ret a + b; }

    assert (#apply(add, [1, 15]) == 16);
}