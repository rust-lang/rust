// run-pass
#![feature(arbitrary_enum_discriminant, const_raw_ptr_deref, test)]

extern crate test;

use test::black_box;

#[allow(dead_code)]
#[repr(u8)]
enum Enum {
    Unit = 3,
    Tuple(u16) = 2,
    Struct {
        a: u8,
        b: u16,
    } = 1,
}

impl Enum {
    const unsafe fn tag(&self) -> u8 {
        *(self as *const Self as *const u8)
    }
}

#[allow(dead_code)]
#[repr(u8)]
enum FieldlessEnum {
    Unit = 3,
    Tuple() = 2,
    Struct {} = 1,
}

fn main() {
    const UNIT: Enum = Enum::Unit;
    const TUPLE: Enum = Enum::Tuple(5);
    const STRUCT: Enum = Enum::Struct{a: 7, b: 11};

    // Ensure discriminants are correct during runtime execution
    assert_eq!(3, unsafe { black_box(UNIT).tag() });
    assert_eq!(2, unsafe { black_box(TUPLE).tag() });
    assert_eq!(1, unsafe { black_box(STRUCT).tag() });

    // Ensure discriminants are correct during CTFE
    const UNIT_TAG: u8 = unsafe { UNIT.tag() };
    const TUPLE_TAG: u8 = unsafe { TUPLE.tag() };
    const STRUCT_TAG: u8 = unsafe { STRUCT.tag() };

    assert_eq!(3, UNIT_TAG);
    assert_eq!(2, TUPLE_TAG);
    assert_eq!(1, STRUCT_TAG);

    // Ensure `as` conversions are correct
    assert_eq!(3, FieldlessEnum::Unit as u8);
    assert_eq!(2, FieldlessEnum::Tuple() as u8);
    assert_eq!(1, FieldlessEnum::Struct{} as u8);
}
