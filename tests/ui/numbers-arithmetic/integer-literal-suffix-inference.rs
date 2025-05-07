//@ run-pass

pub fn main() {
    fn id_i8(n: i8) -> i8 { n }
    fn id_i16(n: i16) -> i16 { n }
    fn id_i32(n: i32) -> i32 { n }
    fn id_i64(n: i64) -> i64 { n }

    fn id_uint(n: usize) -> usize { n }
    fn id_u8(n: u8) -> u8 { n }
    fn id_u16(n: u16) -> u16 { n }
    fn id_u32(n: u32) -> u32 { n }
    fn id_u64(n: u64) -> u64 { n }

    let _i: i8 = -128;
    let j = -128;
    id_i8(j);
    id_i8(-128);

    let _i: i16 = -32_768;
    let j = -32_768;
    id_i16(j);
    id_i16(-32_768);

    let _i: i32 = -2_147_483_648;
    let j = -2_147_483_648;
    id_i32(j);
    id_i32(-2_147_483_648);

    let _i: i64 = -9_223_372_036_854_775_808;
    let j = -9_223_372_036_854_775_808;
    id_i64(j);
    id_i64(-9_223_372_036_854_775_808);

    let _i: usize = 1;
    let j = 1;
    id_uint(j);
    id_uint(1);

    let _i: u8 = 255;
    let j = 255;
    id_u8(j);
    id_u8(255);

    let _i: u16 = 65_535;
    let j = 65_535;
    id_u16(j);
    id_u16(65_535);

    let _i: u32 = 4_294_967_295;
    let j = 4_294_967_295;
    id_u32(j);
    id_u32(4_294_967_295);

    let _i: u64 = 18_446_744_073_709_551_615;
    let j = 18_446_744_073_709_551_615;
    id_u64(j);
    id_u64(18_446_744_073_709_551_615);
}
