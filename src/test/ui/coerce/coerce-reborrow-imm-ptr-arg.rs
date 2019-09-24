// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
