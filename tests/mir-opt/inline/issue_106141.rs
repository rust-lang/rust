// ignore-wasm32 compiled with panic=abort by default
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
