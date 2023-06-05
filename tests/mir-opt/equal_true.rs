// unit-test InstSimplify

// EMIT_MIR equal_true.opt.InstSimplify.diff

fn opt(x: bool) -> i32 {
    if x == true { 0 } else { 1 }
}

fn main() {
    opt(true);
}
