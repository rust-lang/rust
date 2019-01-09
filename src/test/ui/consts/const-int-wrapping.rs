fn main() {
    let x: &'static i32 = &(5_i32.wrapping_add(3)); //~ ERROR does not live long enough
    let y: &'static i32 = &(5_i32.wrapping_sub(3)); //~ ERROR does not live long enough
    let z: &'static i32 = &(5_i32.wrapping_mul(3)); //~ ERROR does not live long enough
    let a: &'static i32 = &(5_i32.wrapping_shl(3)); //~ ERROR does not live long enough
    let b: &'static i32 = &(5_i32.wrapping_shr(3)); //~ ERROR does not live long enough
}
