// Check that conversions by reference which are not lossless are not implemented.

fn main () {
    // larger, signed
    let _: i8 = From::from(&1_i16); //~ ERROR
    let _: i16 = From::from(&1_i32); //~ ERROR
    let _: i32 = From::from(&1_i64); //~ ERROR
    let _: i64 = From::from(&1_i128); //~ ERROR

    // larger, unsigned
    let _: u8 = From::from(&1_u16); //~ ERROR
    let _: u16 = From::from(&1_u32); //~ ERROR
    let _: u32 = From::from(&1_u64); //~ ERROR
    let _: u64 = From::from(&1_u128); //~ ERROR

    // mixed signs
    let _: i8 = From::from(&1_u8); //~ ERROR
    let _: u16 = From::from(&1_i8); //~ ERROR
    let _: i32 = From::from(&1_u32); //~ ERROR
    let _: u64 = From::from(&1_i32); //~ ERROR
    let _: i128 = From::from(&1_u128); //~ ERROR
}
