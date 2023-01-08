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
    // DestProp may unify `index` with `_0`.  Check that inlining does not ICE.
    if buffer[index] {
        index
    } else {
        0
    }
}

fn main() {
    let _ = outer();
}
