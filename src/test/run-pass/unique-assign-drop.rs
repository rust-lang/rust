fn main() {
    let i = ~1;
    let j = ~2;
    // Should drop the previous value of j
    j = i;
    assert *j == 1;
}