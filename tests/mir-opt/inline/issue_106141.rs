// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
pub fn outer() -> usize {
    inner()
}

fn index() -> usize {
    loop {}
}

#[inline]
fn inner() -> usize {
    let buffer = &[true];
    let index = index();
    if buffer[index] {
        index
    } else {
        0
    }
}

fn main() {
    outer();
}

// EMIT_MIR issue_106141.outer.Inline.diff
