fn main() {
    match Some("foo") {
        None::<isize> => {}   //~ ERROR mismatched types
        Some(_) => {}
    }
}
