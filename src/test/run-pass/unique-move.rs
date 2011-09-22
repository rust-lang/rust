fn main() {
    let i = ~100;
    let j;
    j <- i;
    assert *j == 100;
}