//@ run-pass
#![allow(dead_code)]

fn sum(x: &[isize]) -> isize {
    let mut sum = 0;
    for y in x { sum += *y; }
    return sum;
}

fn sum_mut(y: &mut [isize]) -> isize {
    sum(y)
}

fn sum_imm(y: &[isize]) -> isize {
    sum(y)
}

pub fn main() {}
