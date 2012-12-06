fn main() {
    let i = ~mut 1;
    // Should be a copy
    let mut j;
    j = copy i;
    *i = 2;
    *j = 3;
    assert *i == 2;
    assert *j == 3;
}
