#![allow(dead_code, unused_variables)]

struct Aggregate {
    byte: u8,
    word: u16,
}

struct TupleAggregate(u8, u16);

struct Nested {
    aggregate: Aggregate,
    tuple: (u8, u16),
}

enum Choice {
    Unit,
    Tuple(u8, u16),
    Struct { n: u32, ok: bool },
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
    let tuple = (0x01_u8, 0x0203_u16);
    let tuple_struct = TupleAggregate(0x04_u8, 0x0506_u16);
    let array = [0x07_u8, 0x08, 0x09];
    let nested = Nested { aggregate: Aggregate { byte: 0x0a, word: 0x0b0c }, tuple };
    let unit_variant = Choice::Unit;
    let tuple_variant = Choice::Tuple(0x0d, 0x0e0f);
    let struct_variant = Choice::Struct { n: 0x10111213, ok: true };
    // Immediate::Uninit.
    let uninit_scalar: u32;

    std::hint::black_box((
        scalar,
        scalar_pointer,
        scalar_pair.len(),
        &mplace,
        &tuple,
        &tuple_struct,
        &array,
        &nested,
        &unit_variant,
        &tuple_variant,
        &struct_variant,
    ));
}
