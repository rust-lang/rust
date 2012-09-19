fn main() {
    let x = ~[10, 20, 30];
    let mut sum = 0;
    for x.each |x| { sum += *x; }
    assert (sum == 60);
}
