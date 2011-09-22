fn main() {
    let i = ~100;
    let j = ~200;
    let j <- i;
    assert *j == 100;
}