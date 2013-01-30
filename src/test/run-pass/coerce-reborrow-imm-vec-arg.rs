pure fn sum(x: &[int]) -> int {
    let mut sum = 0;
    for x.each |y| { sum += *y; }
    return sum;
}

fn sum_mut(y: &mut [int]) -> int {
    sum(y)
}

fn sum_imm(y: &[int]) -> int {
    sum(y)
}

fn sum_const(y: &[const int]) -> int {
    sum(y)
}

fn main() {}