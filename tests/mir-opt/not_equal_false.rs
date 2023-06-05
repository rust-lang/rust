// unit-test: InstSimplify
// EMIT_MIR not_equal_false.opt.InstSimplify.diff

fn opt(x: bool) -> u32 {
    if x != false { 0 } else { 1 }
}

fn main() {
    opt(false);
}
