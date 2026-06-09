//@ run-pass
#![allow(dead_code)]

fn negate(x: &isize) -> isize {
    -*x
}

fn negate_mut(y: &mut isize) -> isize {
    negate(y)
}

fn negate_imm(y: &isize) -> isize {
    negate(y)
}

pub fn main() {}
