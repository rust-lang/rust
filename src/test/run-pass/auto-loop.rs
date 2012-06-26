fn main() {
    let mut sum = 0;
    for vec::each([1, 2, 3, 4, 5]/~) {|x| sum += x; }
    assert (sum == 15);
}
