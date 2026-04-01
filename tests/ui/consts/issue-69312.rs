//@ build-pass

// Verify that the compiler doesn't ICE during const prop while evaluating the index operation.

#![allow(unconditional_panic)]

fn main() {
    let cols = [0u32; 0];
    cols[0];
}
