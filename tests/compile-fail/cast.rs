#![feature(plugin)]
#![plugin(clippy)]

#[deny(cast_precision_loss, cast_possible_truncation, cast_sign_loss, cast_possible_wrap)]
fn main() {
    // Test cast_precision_loss
    1i32 as f32; //~ERROR casting i32 to f32 causes a loss of precision (i32 is 32 bits wide, but f32's mantissa is only 23 bits wide)
    1i64 as f32; //~ERROR casting i64 to f32 causes a loss of precision (i64 is 64 bits wide, but f32's mantissa is only 23 bits wide)
    1i64 as f64; //~ERROR casting i64 to f64 causes a loss of precision (i64 is 64 bits wide, but f64's mantissa is only 52 bits wide)
    1u32 as f32; //~ERROR casting u32 to f32 causes a loss of precision (u32 is 32 bits wide, but f32's mantissa is only 23 bits wide)
    1u64 as f32; //~ERROR casting u64 to f32 causes a loss of precision (u64 is 64 bits wide, but f32's mantissa is only 23 bits wide)
    1u64 as f64; //~ERROR casting u64 to f64 causes a loss of precision (u64 is 64 bits wide, but f64's mantissa is only 52 bits wide)
    1i32 as f64; // Should not trigger the lint
    1u32 as f64; // Should not trigger the lint

    // Test cast_possible_truncation
    1f32 as i32;   //~ERROR casting f32 to i32 may truncate the value
    1f32 as u32;   //~ERROR casting f32 to u32 may truncate the value
                  //~^ERROR casting f32 to u32 may lose the sign of the value
    1f64 as f32;   //~ERROR casting f64 to f32 may truncate the value
    1i32 as i8;    //~ERROR casting i32 to i8 may truncate the value
    1i32 as u8;    //~ERROR casting i32 to u8 may truncate the value
                  //~^ERROR casting i32 to u8 may lose the sign of the value
    1f64 as isize; //~ERROR casting f64 to isize may truncate the value
    1f64 as usize; //~ERROR casting f64 to usize may truncate the value
                  //~^ERROR casting f64 to usize may lose the sign of the value

    // Test cast_possible_wrap
    1u8 as i8;       //~ERROR casting u8 to i8 may wrap around the value
    1u16 as i16;     //~ERROR casting u16 to i16 may wrap around the value
    1u32 as i32;     //~ERROR casting u32 to i32 may wrap around the value
    1u64 as i64;     //~ERROR casting u64 to i64 may wrap around the value
    1usize as isize; //~ERROR casting usize to isize may wrap around the value

    // Test cast_sign_loss
    1i32 as u32;     //~ERROR casting i32 to u32 may lose the sign of the value
    1isize as usize; //~ERROR casting isize to usize may lose the sign of the value

    // Extra checks for *size
    // Casting from *size
    1isize as i8;  //~ERROR casting isize to i8 may truncate the value
    1isize as f64; //~ERROR casting isize to f64 causes a loss of precision on targets with 64-bit wide pointers (isize is 64 bits wide, but f64's mantissa is only 52 bits wide)
    1usize as f64; //~ERROR casting usize to f64 causes a loss of precision on targets with 64-bit wide pointers (usize is 64 bits wide, but f64's mantissa is only 52 bits wide)
    1isize as f32; //~ERROR casting isize to f32 causes a loss of precision (isize is 32 or 64 bits wide, but f32's mantissa is only 23 bits wide)
    1usize as f32; //~ERROR casting usize to f32 causes a loss of precision (usize is 32 or 64 bits wide, but f32's mantissa is only 23 bits wide)
    1isize as i32; //~ERROR casting isize to i32 may truncate the value on targets with 64-bit wide pointers
    1isize as u32; //~ERROR casting isize to u32 may lose the sign of the value
                  //~^ERROR casting isize to u32 may truncate the value on targets with 64-bit wide pointers
    1usize as u32; //~ERROR casting usize to u32 may truncate the value on targets with 64-bit wide pointers
    1usize as i32; //~ERROR casting usize to i32 may truncate the value on targets with 64-bit wide pointers
                  //~^ERROR casting usize to i32 may wrap around the value on targets with 32-bit wide pointers
    // Casting to *size
    1i64 as isize; //~ERROR casting i64 to isize may truncate the value on targets with 32-bit wide pointers
    1i64 as usize; //~ERROR casting i64 to usize may truncate the value on targets with 32-bit wide pointers
                  //~^ERROR casting i64 to usize may lose the sign of the value
    1u64 as isize; //~ERROR casting u64 to isize may truncate the value on targets with 32-bit wide pointers
                  //~^ERROR casting u64 to isize may wrap around the value on targets with 64-bit wide pointers
    1u64 as usize; //~ERROR casting u64 to usize may truncate the value on targets with 32-bit wide pointers
    1u32 as isize; //~ERROR casting u32 to isize may wrap around the value on targets with 32-bit wide pointers
    1u32 as usize; // Should not trigger any lint
    1i32 as isize; // Neither should this
    1i32 as usize; //~ERROR casting i32 to usize may lose the sign of the value
}
