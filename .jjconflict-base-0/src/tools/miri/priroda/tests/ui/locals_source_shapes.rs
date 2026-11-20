#![allow(dead_code, unused_variables)]

struct Named {
    a: u8,
    b: u16,
}

struct EmptyBraced {}

struct UnitStruct;

struct TupleStruct(u8, u16);

struct SingleTupleStruct(u8);

struct ZeroTupleStruct();

enum Variants {
    Unit,
    Tuple(u8, u16),
    SingleTuple(u8),
    Struct { n: u32, ok: bool },
    EmptyStruct {},
}

union RawUnion {
    byte: u8,
    word: u16,
}

fn main() {
    let named = Named { a: 0x01, b: 0x0302 };
    let empty_braced = EmptyBraced {};
    let unit_struct = UnitStruct;
    let tuple_struct = TupleStruct(0x04, 0x0605);
    let single_tuple_struct = SingleTupleStruct(0x07);
    let zero_tuple_struct = ZeroTupleStruct();
    let tuple_zero = ();
    let tuple_one = (0x08_u8,);
    let tuple_many = (0x09_u8, 0x0b0a_u16);
    let array_zero: [u8; 0] = [];
    let array_one = [0x0c_u8];
    let array_many = [0x0d_u8, 0x0e, 0x0f];
    let variant_unit = Variants::Unit;
    let variant_tuple = Variants::Tuple(0x10, 0x1211);
    let variant_single_tuple = Variants::SingleTuple(0x13);
    let variant_struct = Variants::Struct { n: 0x17161514, ok: false };
    let variant_empty_struct = Variants::EmptyStruct {};
    let raw_union = RawUnion { word: 0x1918 };

    std::hint::black_box((
        &named,
        &empty_braced,
        &unit_struct,
        &tuple_struct,
        &single_tuple_struct,
        &zero_tuple_struct,
        &tuple_zero,
        &tuple_one,
        &tuple_many,
        &array_zero,
        &array_one,
        &array_many,
        &variant_unit,
        &variant_tuple,
        &variant_single_tuple,
        &variant_struct,
        &variant_empty_struct,
        &raw_union,
    ));
    std::hint::black_box(0_u8);
}
