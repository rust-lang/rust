#![feature(plugin)]
#![plugin(clippy)]

#[deny(cast_precision_loss, cast_possible_truncation, cast_sign_loss)]
#[allow(dead_code)]
fn main() {
    let i : i32 = 42;
    let u : u32 = 42;
    let f : f32 = 42.0;

    // Test cast_precision_loss
    i as f32; //~ERROR converting from i32 to f32, which causes a loss of precision (i32 is 32 bits wide, but f32's mantissa is only 23 bits wide)
    (i as i64) as f32; //~ERROR converting from i64 to f32, which causes a loss of precision (i64 is 64 bits wide, but f32's mantissa is only 23 bits wide)
    (i as i64) as f64; //~ERROR converting from i64 to f64, which causes a loss of precision (i64 is 64 bits wide, but f64's mantissa is only 52 bits wide)
    u as f32; //~ERROR converting from u32 to f32, which causes a loss of precision (u32 is 32 bits wide, but f32's mantissa is only 23 bits wide)
    (u as u64) as f32; //~ERROR converting from u64 to f32, which causes a loss of precision (u64 is 64 bits wide, but f32's mantissa is only 23 bits wide)
    (u as u64) as f64; //~ERROR converting from u64 to f64, which causes a loss of precision (u64 is 64 bits wide, but f64's mantissa is only 52 bits wide)
    i as f64; // Should not trigger the lint
    u as f64; // Should not trigger the lint

    // Test cast_possible_truncation
    f as i32; //~ERROR casting f32 to i32 may cause truncation of the value
    f as u32; //~ERROR casting f32 to u32 may cause truncation of the value
              //~^ERROR casting from f32 to u32 loses the sign of the value
    i as u8;  //~ERROR casting i32 to u8 may cause truncation of the value
              //~^ERROR casting from i32 to u8 loses the sign of the value
    (f as f64) as f32; //~ERROR casting f64 to f32 may cause truncation of the value
    i as i8;  //~ERROR casting i32 to i8 may cause truncation of the value
    u as i32; //~ERROR casting u32 to i32 may cause truncation of the value

    // Test cast_sign_loss
    i as u32; //~ERROR casting from i32 to u32 loses the sign of the value

    // Extra checks for usize/isize
    /*
    let is : isize = -42;
    let us : usize = 42;
    is as usize; //ERROR casting from isize to usize loses the sign of the value
    is as i8; //ERROR casting isize to i8 may cause truncation of the value
    is as f64; //ERROR converting from isize to f64, which causes a loss of precision on 64-bit architectures (isize is 64 bits wide, but f64's mantissa is only 52 bits wide)
    us as f64; //ERROR converting from usize to f64, which causes a loss of precision on 64-bit architectures (usize is 64 bits wide, but f64's mantissa is only 52 bits wide)
    */
}