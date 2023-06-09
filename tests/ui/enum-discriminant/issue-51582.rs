// run-pass
#![feature(core_intrinsics)]

#[repr(i8)]
pub enum Enum {
    VariantA,
    VariantB,
}

fn make_b() -> Enum { Enum::VariantB }

fn main() {
    assert_eq!(1, make_b() as i8);
    assert_eq!(1, make_b() as u8);
    assert_eq!(1, make_b() as i32);
    assert_eq!(1, make_b() as u32);
    assert_eq!(1, std::intrinsics::discriminant_value(&make_b()));
}
