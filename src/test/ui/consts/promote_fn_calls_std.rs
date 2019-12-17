// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let x: &'static u8 = &u8::MAX;
    let x: &'static u16 = &u16::MAX;
    let x: &'static u32 = &u32::MAX;
    let x: &'static u64 = &u64::MAX;
    let x: &'static u128 = &u128::MAX;
    let x: &'static usize = &usize::MAX;
    let x: &'static u8 = &u8::MIN;
    let x: &'static u16 = &u16::MIN;
    let x: &'static u32 = &u32::MIN;
    let x: &'static u64 = &u64::MIN;
    let x: &'static u128 = &u128::MIN;
    let x: &'static usize = &usize::MIN;
    let x: &'static i8 = &i8::MAX;
    let x: &'static i16 = &i16::MAX;
    let x: &'static i32 = &i32::MAX;
    let x: &'static i64 = &i64::MAX;
    let x: &'static i128 = &i128::MAX;
    let x: &'static isize = &isize::MAX;
    let x: &'static i8 = &i8::MIN;
    let x: &'static i16 = &i16::MIN;
    let x: &'static i32 = &i32::MIN;
    let x: &'static i64 = &i64::MIN;
    let x: &'static i128 = &i128::MIN;
    let x: &'static isize = &isize::MIN;
}
