#![feature(plugin)]
#![plugin(clippy)]

#[deny(cast_precision_loss, cast_possible_overflow, cast_sign_loss)]
fn main() {
    let i : i32 = 42;
    let u : u32 = 42;
    let f : f32 = 42.0;

    // Test cast_precision_loss
    i as f32; //~ERROR converting from i32 to f32, which causes a loss of precision
    (i as i64) as f32; //~ERROR converting from i64 to f32, which causes a loss of precision
    (i as i64) as f64; //~ERROR converting from i64 to f64, which causes a loss of precision
    u as f32; //~ERROR converting from u32 to f32, which causes a loss of precision
    (u as u64) as f32; //~ERROR converting from u64 to f32, which causes a loss of precision
    (u as u64) as f64; //~ERROR converting from u64 to f64, which causes a loss of precision
    i as f64; // Should not trigger the lint
    u as f64; // Should not trigger the lint
    
    // Test cast_possible_overflow
    f as i32; //~ERROR the contents of a f32 may overflow a i32
    f as u32; //~ERROR the contents of a f32 may overflow a u32
              //~^ERROR casting from f32 to u32 loses the sign of the value
    i as u8;  //~ERROR the contents of a i32 may overflow a u8
              //~^ERROR casting from i32 to u8 loses the sign of the value
    (f as f64) as f32; //~ERROR the contents of a f64 may overflow a f32
    i as i8;  //~ERROR the contents of a i32 may overflow a i8
    
    // Test cast_sign_loss
    i as u32; //~ERROR casting from i32 to u32 loses the sign of the value
}