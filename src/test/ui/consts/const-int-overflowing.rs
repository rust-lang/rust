fn main() {
    let x: &'static (i32, bool) = &(5_i32.overflowing_add(3)); //~ ERROR does not live long enough
    let y: &'static (i32, bool) = &(5_i32.overflowing_sub(3)); //~ ERROR does not live long enough
    let z: &'static (i32, bool) = &(5_i32.overflowing_mul(3)); //~ ERROR does not live long enough
}
