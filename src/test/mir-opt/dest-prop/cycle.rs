//! Tests that cyclic assignments don't hang DestinationPropagation, and result in reasonable code.

fn val() -> i32 {
    1
}

// EMIT_MIR rustc.main.DestinationPropagation.diff
fn main() {
    let mut x = val();
    let y = x;
    let z = y;
    x = z;

    drop(x);
}
