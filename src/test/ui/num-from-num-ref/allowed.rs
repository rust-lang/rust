// check-pass

fn main () {
    // same size, signed
    let _: i8 = From::from(&1_i8);
    let _: i16 = From::from(&1_i16);
    let _: i32 = From::from(&1_i32);
    let _: i64 = From::from(&1_i64);
    let _: i128 = From::from(&1_i128);

    // same size, unsigned
    let _: u8 = From::from(&1_u8);
    let _: u16 = From::from(&1_u16);
    let _: u32 = From::from(&1_u32);
    let _: u64 = From::from(&1_u64);
    let _: u128 = From::from(&1_u128);

    // smaller, signed
    let _: i16 = From::from(&1_i8);
    let _: i32 = From::from(&1_i16);
    let _: i64 = From::from(&1_i32);
    let _: i128 = From::from(&1_i64);

    // smaller, unsigned
    let _: u16 = From::from(&1_u8);
    let _: u32 = From::from(&1_u16);
    let _: u64 = From::from(&1_u32);
    let _: u128 = From::from(&1_u64);

    // mixed signs
    let _: i16 = From::from(&1_u8);
    let _: i32 = From::from(&1_u16);
    let _: i64 = From::from(&1_u32);
    let _: i128 = From::from(&1_u64);
}
