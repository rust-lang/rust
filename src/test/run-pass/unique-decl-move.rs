fn main() {
    let i = ~100;
    let j <- i;
    assert *j == 100;
}