fn main() {
    let x = ~[10, 20, 30];
    let sum = 0;
    for x in x { sum += x; }
    assert (sum == 60);
}
