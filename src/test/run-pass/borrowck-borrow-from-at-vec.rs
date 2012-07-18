fn sum_slice(x: &[int]) -> int {
    let mut sum = 0;
    for x.each |i| { sum += i; }
    ret sum;
}

fn main() {
    let x = @[1, 2, 3];
    assert sum_slice(x) == 6;
}