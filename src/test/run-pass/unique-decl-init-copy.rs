fn main() {
    let i = ~mut 1;
    // Should be a copy
    let j = i;
    *i = 2;
    *j = 3;
    assert *i == 2;
    assert *j == 3;
}