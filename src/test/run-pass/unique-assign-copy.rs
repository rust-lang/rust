fn main() {
    let i = ~mutable 1;
    // Should be a copy
    let j;
    j = i;
    *i = 2;
    *j = 3;
    assert *i == 2;
    assert *j == 3;
}