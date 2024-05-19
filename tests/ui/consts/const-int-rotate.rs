fn main() {
    let x: &'static i32 = &(5_i32.rotate_left(3));
    //~^ ERROR temporary value dropped while borrowed
    let y: &'static i32 = &(5_i32.rotate_right(3));
    //~^ ERROR temporary value dropped while borrowed
}
