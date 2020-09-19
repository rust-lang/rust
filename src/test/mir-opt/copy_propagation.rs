// compile-flags: -Zunsound-mir-opts
// EMIT_MIR copy_propagation.test.CopyPropagation.diff

fn test(x: u32) -> u32 {
    let y = x;
    y
}

fn main() {
    // Make sure the function actually gets instantiated.
    test(0);
}
