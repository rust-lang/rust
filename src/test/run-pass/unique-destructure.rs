fn main() {
    let ~{a, b} = ~{a: 100, b: 200};
    assert a + b == 300;
}