fn main() {
    let _ = [0; (&0 as *const i32) as usize]; //~ ERROR casting pointers to integers in constants
}
