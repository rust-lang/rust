fn main() {
    match Some(vec![3]) {
        Some(vec![x]) => (), //~ ERROR unexpected `(` after qualified path
        _ => (),
    }
}
