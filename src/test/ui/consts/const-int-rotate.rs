fn main() {
    let x: &'static i32 = &(5_i32.rotate_left(3)); //~ ERROR does not live long enough
    let y: &'static i32 = &(5_i32.rotate_right(3)); //~ ERROR does not live long enough
}
