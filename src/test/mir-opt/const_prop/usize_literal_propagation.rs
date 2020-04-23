// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let x = 1;
    consume(x);
}

#[inline(never)]
fn consume(_: usize) { }
