fn main() {
    let i = ~100;
    let j = move i;
    assert *j == 100;
}