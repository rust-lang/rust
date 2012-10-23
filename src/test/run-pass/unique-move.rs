fn main() {
    let i = ~100;
    let mut j;
    j = move i;
    assert *j == 100;
}