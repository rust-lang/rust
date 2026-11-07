// Verify that we do not warn on fields that are part of transparent types.
//@ check-pass
#![deny(dead_code)]

#[repr(transparent)]
struct NamedStruct { field: u8 }

#[repr(transparent)]
struct TupleStruct(u8);

fn main() {
    let _ = NamedStruct { field: 1 };
    let _ = TupleStruct(1);
}
