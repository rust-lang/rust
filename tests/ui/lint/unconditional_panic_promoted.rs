//@ build-fail

fn main() {
    // MIR encodes this as a reborrow from a promoted constant.
    // But the array length can still be gotten from the type.
    let slice = &[0, 1];
    let _ = slice[2]; //~ ERROR: this operation will panic at runtime [unconditional_panic]
}
