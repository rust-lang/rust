fn main() {
    let sum = 0;
    for x  in ~[1, 2, 3, 4, 5] { sum += x; }
    assert (sum == 15);
}