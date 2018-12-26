fn main() {
    let x: &'static bool = &(5_i32.is_negative()); //~ ERROR does not live long enough
    let y: &'static bool = &(5_i32.is_positive()); //~ ERROR does not live long enough
}
