pure fn negate(x: &int) -> int {
    -*x
}

fn negate_mut(y: &mut int) -> int {
    negate(y)
}

fn negate_imm(y: &int) -> int {
    negate(y)
}

fn negate_const(y: &const int) -> int {
    negate(y)
}

fn main() {}
