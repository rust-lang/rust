#![allow(dead_code, unused_variables)]

struct Aggregate {
    byte: u8,
    word: u16,
}

fn main() {
    // Edge case for `None` from `as_mplace_or_imm`: moved/drop temporaries become dead.
    let dead_box = Box::new(0x11_u8);
    let consumed = dead_box;
    drop(consumed);

    let pointed_box = Box::new(0x11_u8);
    let pointer_box = &pointed_box;

    // Immediate::Scalar.
    let scalar = 0x2a_u8;
    let pointed = 0x33_u8;
    // Immediate::Scalar with pointer provenance.
    let scalar_pointer = &pointed;
    // Immediate::ScalarPair.
    let scalar_pair = &[10_u8, 20_u8][..];
    // Either::Left mplace/indirect storage.
    let mplace = Aggregate { byte: scalar, word: 0x1234 };
    // Immediate::Uninit.
    let uninit_scalar: u32;

    std::hint::black_box((scalar, scalar_pointer, scalar_pair.len(), &mplace));
}
